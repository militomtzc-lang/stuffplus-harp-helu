"""stuffplus_eval.py — evaluaciones de la sesión del 8 oct 2026 (versionado de celdas de notebook).

Contenido (todo asociativo, nada causal):
  1. verify_sources        regla del 7 oct: `d` debe reproducir el `y` de un OOF existente.
  2. tercile_level_fourseam  pendiente 1: nivel de Stuff por tercil de IVB (Four-Seam), No vs Extreme.
  3. changeup_h2           pendiente 2: Spearman por pitcher-año del Changeup (un solo intento).
  4. ALE con forma transportada (entregable 2): check_bundles, run_ale, plot_curves.

Requisitos:
  * `d` = DataFrame de trabajo (frame_input.parquet + build_frame), el mismo que usó el reentrenamiento.
  * stuffplus_retrain.py en la misma carpeta (se importa como `sr`).
  * Modelos finales en retrain_oct2026/models/final__stuff__{whiff_swing,gb_bip}.joblib
  * oof_full/pitcher__{capa}__{target}.parquet (trae y, in_mask, p; NO trae base_count).

Limitaciones declaradas (ver handoff_addendum_8oct2026_parte2.md):
  * IC de bootstrap = solo muestreo de pitchers. No cubre error del modelo ni del delta de forma.
  * Los modelos finales se evalúan sobre filas de su propio entrenamiento (AUC in-sample ~0.65 vs OOF ~0.62-0.64).
  * Transporte de forma: IVB x0.78 en FB (22% medido en Four-Seam, extrapolado a Sinker/Cutter);
    IVB aditivo por tipo en BRK; HB_arm aditivo por tipo. Velocidad y spin NO se transportan.
  * "No Altitude" no es nivel del mar; "Extreme" agrupa 5-6 parques.
  * Sweeper excluido de todo. Cutter sin ALE propio. Curveball y Slider-gb: solo tabla, curvas sin revisar.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import stuffplus_retrain as sr

SEED = 42
RT = Path("resultados_stuffplus/retrain_oct2026")
E1 = Path("resultados_stuffplus/entregable1_v2")
E2 = Path("resultados_stuffplus/entregable2_v2")
EXT, NO = "Extreme Altitude", "No Altitude"

HEADS = {"whiff_swing": "whiff|swing", "gb_bip": "groundball|bola en juego"}
FEATS = ["RelSpeed", "SpinRate", "InducedVertBreak", "HB_arm", "RelHeight", "RelSide_arm", "Extension"]
FAM_TYPES = {"FB": ["Four-Seam", "Sinker", "Cutter"], "BRK": ["Slider", "Curveball"]}
TYPES = {"Four-Seam": ["Four-Seam"], "Sinker": ["Sinker"], "Slider": ["Slider"], "Curveball": ["Curveball"]}

# Delta observado Extreme - No dentro del mismo pitcher (8 oct): (dIVB, dHB_arm) en pulgadas
DELTA = {"Four-Seam": (-3.80, -2.75), "Sinker": (-2.22, -4.31), "Cutter": (-2.37, -0.32),
         "Curveball": (1.78, 3.24), "Slider": (-0.72, 1.19)}
IVB_FACTOR = {"Four-Seam": 0.78, "Sinker": 0.78, "Cutter": 0.78}   # proporcional (~22%)
IVB_ADD = {"Slider": -0.72, "Curveball": 1.78}                       # aditivo por tipo
HB_ADD = {t: v[1] for t, v in DELTA.items()}


def load_oof(layer: str, target: str, scheme: str = "pitcher") -> pd.DataFrame:
    return pd.read_parquet(RT / "oof_full" / f"{scheme}__{layer}__{target}.parquet")


# ----------------------------------------------------------------------------
# 1. Verificación de fuentes
# ----------------------------------------------------------------------------
def verify_sources(d: pd.DataFrame) -> None:
    """Falla si `d` no reproduce el target `rv` del OOF (evita comparar targets distintos sin error)."""
    assert d.PitchUID.is_unique
    o_rv = load_oof("stuff", "rv")
    _, y_ref, _ = sr.make_targets(d)["rv"]
    ref = pd.Series(y_ref, index=d.PitchUID)
    a = o_rv.set_index("PitchUID")["y"]
    b = ref.reindex(a.index)
    ok = a.notna() & b.notna()
    assert np.allclose(a[ok], b[ok], atol=1e-9), "d NO coincide con los OOF"
    print("fuente OK:", int(ok.sum()), "filas verificadas")


# ----------------------------------------------------------------------------
# 2. Pendiente 1: nivel por tercil de IVB (Four-Seam)
# ----------------------------------------------------------------------------
def tercile_level_fourseam(d: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """Brecha Alto-Bajo por tercil de IVB base (No Altitude), en No y Extreme, bootstrap por pitcher.
    pred100 = 100*p del cabezal directo de rv (Stuff), + = peor. obs100 = solo chequeo de consistencia.
    stuff_park = misma info que pred100 en otra escala: NO es segunda evidencia."""
    rng = np.random.default_rng(SEED)
    o_rv = load_oof("stuff", "rv")
    bc = pd.read_parquet(RT / "oof" / "pitcher__stuff__rv.parquet", columns=["PitchUID", "base_count"])
    sc = pd.read_parquet(E1 / "pitch_scores.parquet", columns=["PitchUID", "stuff_park"])
    fb = d.loc[d.AutoPitchType.eq("Four-Seam") & d.altitude_category.isin([NO, EXT]),
               ["PitchUID", "pitcher_anon_id", "altitude_category", "InducedVertBreak"]]
    x = (fb.merge(o_rv[["PitchUID", "y", "p"]], on="PitchUID", how="left")
           .merge(bc, on="PitchUID", how="left")
           .merge(sc, on="PitchUID", how="left"))
    print("cobertura base_count (esperado ~0.97):", round(x.base_count.notna().mean(), 3))
    x["pred100"] = 100 * x["p"]
    x["obs100"] = 100 * (x["y"] - x["base_count"])
    x["alt"] = np.where(x.altitude_category.eq(EXT), "E", "N")

    cnt = x.groupby(["pitcher_anon_id", "alt"]).size().unstack(fill_value=0)
    elig = cnt[(cnt["E"] >= 30) & (cnt["N"] >= 30)].index          # mismo criterio que 7a
    x = x[x.pitcher_anon_id.isin(elig)].copy()
    ivb_n = x[x.alt == "N"].groupby("pitcher_anon_id").InducedVertBreak.mean()
    terc = pd.qcut(ivb_n, 3, labels=["Bajo", "Medio", "Alto"])
    print("pitchers por tercil:", terc.value_counts().to_dict())
    print("IVB base medio:", ivb_n.groupby(terc, observed=True).mean().round(2).to_dict())

    metrics = ["pred100", "obs100", "stuff_park"]
    tab = x.groupby(["pitcher_anon_id", "alt"])[metrics].agg(["sum", "count"]).unstack("alt")
    P = tab.index.to_numpy()
    T = terc.reindex(P).astype(str).to_numpy()
    A = {(m, st, a): tab[(m, st, a)].fillna(0).to_numpy(float)
         for m in metrics for st in ("sum", "count") for a in ("N", "E")}

    def stats(w):
        r = {}
        for m in metrics:
            for t in ("Bajo", "Medio", "Alto"):
                s = T == t
                for a in ("N", "E"):
                    num = (w[s] * A[(m, "sum", a)][s]).sum()
                    den = max((w[s] * A[(m, "count", a)][s]).sum(), 1e-9)
                    r[(m, t, a)] = num / den
            for a in ("N", "E"):
                r[(m, "BRECHA Alto-Bajo", a)] = r[(m, "Alto", a)] - r[(m, "Bajo", a)]
            r[(m, "CAMBIO de brecha", "E-N")] = r[(m, "BRECHA Alto-Bajo", "E")] - r[(m, "BRECHA Alto-Bajo", "N")]
        return r

    pt = stats(np.ones(len(P)))
    boots = [stats(rng.multinomial(len(P), np.ones(len(P)) / len(P)).astype(float)) for _ in range(n_boot)]
    out = pd.DataFrame({k: {"est": pt[k],
                            "lo": np.percentile([b[k] for b in boots], 2.5),
                            "hi": np.percentile([b[k] for b in boots], 97.5)} for k in pt}).T
    out.index.names = ["metrica", "grupo", "alt"]
    return out


# ----------------------------------------------------------------------------
# 3. Pendiente 2: Changeup (un solo intento; no agregar variantes)
# ----------------------------------------------------------------------------
def changeup_h2(d: pd.DataFrame, n_boot: int = 500) -> tuple[pd.DataFrame, float]:
    """Spearman por pitcher-año (n>=100 pitches, >=30 swings) contra whiff|swing observado.
    H2 solo tendría apoyo si: rho(w_pitch)-rho(w_stuff) tiene IC>0 Y rho(vgap)>0 con IC que excluye 0.
    OJO: eso prueba el MECANISMO (de dónde viene el whiff), no la consecuencia de altitud de H2."""
    rng = np.random.default_rng(SEED)
    ch = d.loc[d.AutoPitchType.eq("Changeup"),
               ["PitchUID", "pitcher_anon_id", "year", "is_swing", "is_whiff", "relspeed_diff_vs_primary"]].copy()
    ch["swing"] = pd.to_numeric(ch.is_swing, errors="coerce")
    ch["whiff"] = pd.to_numeric(ch.is_whiff, errors="coerce")
    ch["vgap"] = -ch["relspeed_diff_vs_primary"]          # mph por debajo del fastball (+ = más brecha)

    scp = pd.read_parquet(E1 / "pitch_scores.parquet",
                          columns=["PitchUID", "stuff_park", "location_park", "pitching_park"])
    hs = load_oof("stuff", "whiff_swing")[["PitchUID", "p"]].rename(columns={"p": "w_stuff"})
    hp = load_oof("pitching", "whiff_swing")[["PitchUID", "p"]].rename(columns={"p": "w_pitch"})
    ch = ch.merge(scp, on="PitchUID", how="left").merge(hs, on="PitchUID", how="left").merge(hp, on="PitchUID", how="left")

    key = ["pitcher_anon_id", "year"]
    allp = ch.groupby(key).agg(n_pitches=("PitchUID", "size"), stuff_park=("stuff_park", "mean"),
                               location_park=("location_park", "mean"), pitching_park=("pitching_park", "mean"),
                               vgap=("vgap", "mean"))
    sw = ch[ch.swing == 1].groupby(key).agg(sw_n=("whiff", "size"), whiff_sw=("whiff", "mean"),
                                             w_stuff=("w_stuff", "mean"), w_pitch=("w_pitch", "mean"))
    T2 = allp.join(sw, how="inner")
    T2 = T2[(T2.n_pitches >= 100) & (T2.sw_n >= 30)].dropna().reset_index()
    print("celdas pitcher-año Changeup:", len(T2), "| pitchers:", T2.pitcher_anon_id.nunique())

    pred = ["stuff_park", "location_park", "pitching_park", "w_stuff", "w_pitch", "vgap"]

    def rho(df):
        r = {c: spearmanr(df[c], df["whiff_sw"])[0] for c in pred}
        r["w_pitch - w_stuff"] = r["w_pitch"] - r["w_stuff"]
        return r

    pt = rho(T2)
    idx = T2.groupby("pitcher_anon_id").indices
    keys = list(idx)
    bs = pd.DataFrame([rho(T2.iloc[np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))])])
                       for _ in range(n_boot)])
    out = pd.DataFrame({"rho": pd.Series(pt), "lo": bs.quantile(.025), "hi": bs.quantile(.975)})

    # Techo: confiabilidad split-half del whiff|swing observado (Spearman-Brown)
    s = ch[ch.swing == 1].copy()
    s["half"] = rng.integers(0, 2, len(s))
    hh = s.groupby(key + ["half"]).whiff.mean().unstack("half").dropna()
    hh = hh[hh.index.isin(T2.set_index(key).index)]
    r = hh[0].corr(hh[1])
    sb = 2 * r / (1 + r)
    print(f"confiabilidad whiff|swing (SB): {sb:.2f} -> rho maximo alcanzable ~ {np.sqrt(max(sb, 0)):.2f}")
    return out, float(sb)


# ----------------------------------------------------------------------------
# 4. ALE con forma transportada (entregable 2)
# ----------------------------------------------------------------------------
def load_bundles() -> tuple[dict, dict]:
    bnd = {h: joblib.load(RT / "models" / f"final__stuff__{h}.joblib") for h in HEADS}
    lev = bnd["whiff_swing"]["levels"]
    assert bnd["gb_bip"]["levels"] == lev
    return bnd, lev


def check_bundles(d: pd.DataFrame, bnd: dict, lev: dict, n: int = 150_000) -> None:
    """AUC in-sample debe ser algo mayor que el OOF (0.635/0.618), no >~0.70; p media ~ y media."""
    rng = np.random.default_rng(SEED)
    T = sr.make_targets(d)
    X_all = sr._design(d, "stuff", False, lev)
    for h in HEADS:
        _, y, m = T[h]
        pool = np.where(m)[0]
        idx = rng.choice(pool, min(n, len(pool)), replace=False)
        p = bnd[h]["model"].predict_proba(X_all.iloc[idx])[:, 1]
        print(f"{h}: n={len(idx):,} | AUC in-sample {roc_auc_score(y[idx], p):.3f} "
              f"| p media {p.mean():.3f} | y media {y[idx].mean():.3f}")


def tr_params(types) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = pd.Series(types).astype(str)            # astype(str): AutoPitchType es categórica
    a = t.map(IVB_FACTOR).fillna(1.0).to_numpy()
    b = t.map(IVB_ADD).fillna(0.0).to_numpy()
    h = t.map(HB_ADD).to_numpy()
    assert not np.isnan(h).any(), "tipo sin delta de HB"
    return a, b, h


def build_base_tr(d: pd.DataFrame, lev: dict, types: list[str], n: int = 60_000):
    """Base = pitches de No Altitude de `types`; la copia a evaluar lleva la forma transportada y etiqueta Extreme."""
    b = d[d.AutoPitchType.isin(types) & d.altitude_category.eq(NO)
          & d.HB_arm.notna() & d.RelSide_arm.notna()].sample(frac=1, random_state=SEED)
    b = b.iloc[:n].copy()
    extra = set(b.AutoPitchType.astype(str).unique()) - set(types)
    assert not extra, f"tipos fuera de la familia: {extra}"
    a, bb, h = tr_params(b.AutoPitchType)
    z = b.copy()
    z["altitude_category"] = EXT
    z["InducedVertBreak"] = a * z["InducedVertBreak"].to_numpy() + bb
    z["HB_arm"] = z["HB_arm"].to_numpy() + h
    return b, sr._design(z, "stuff", False, lev), (a, bb, h)


def ale_curves_tr(model, Xs, origin, par, pit, feat, nbins: int = 18, n_boot: int = 200):
    """ALE de `feat` con malla en unidades de ORIGEN; IVB/HB se transportan al evaluar. IC: bootstrap de pitchers."""
    a, bb, h = par
    x = origin[feat].to_numpy(float)
    edges = np.unique(np.nanquantile(x, np.linspace(0.05, 0.95, nbins + 1)))
    inside = (x >= edges[0]) & (x <= edges[-1])
    Xi, xi, pi = Xs[inside], x[inside], pit[inside]
    k = np.clip(np.searchsorted(edges, xi, side="right") - 1, 0, len(edges) - 2)
    lo, hi = Xi.copy(), Xi.copy()
    if feat == "InducedVertBreak":
        lo[feat] = a[inside] * edges[k] + bb[inside]
        hi[feat] = a[inside] * edges[k + 1] + bb[inside]
    elif feat == "HB_arm":
        lo[feat] = edges[k] + h[inside]
        hi[feat] = edges[k + 1] + h[inside]
    else:
        lo[feat], hi[feat] = edges[k], edges[k + 1]
    diff = model.predict_proba(hi)[:, 1] - model.predict_proba(lo)[:, 1]
    nb = len(edges) - 1

    def curve(w):
        num = np.bincount(k, weights=w * diff, minlength=nb)
        den = np.bincount(k, weights=w, minlength=nb)
        eff = np.divide(num, den, out=np.zeros(nb), where=den > 0)
        ale = np.concatenate([[0.0], np.cumsum(eff)])
        return ale - np.average(0.5 * (ale[k] + ale[k + 1]), weights=w)

    pt = curve(np.ones(len(k)))
    codes, uniq = pd.factorize(pi)
    r = np.random.default_rng(SEED)
    bt = np.array([curve(r.multinomial(len(uniq), np.ones(len(uniq)) / len(uniq))[codes].astype(float))
                   for _ in range(n_boot)])
    return edges, 100 * pt, 100 * np.percentile(bt, 2.5, 0), 100 * np.percentile(bt, 97.5, 0), 100 * bt


def iqr_effect(e, c, x_vals) -> float:
    """Efecto p25 -> p75 (más representativo que p5-p95). OJO: puede esconder saturaciones; ver la curva."""
    q25, q75 = np.nanpercentile(x_vals, [25, 75])
    return float(np.interp(q75, e, c) - np.interp(q25, e, c))


def run_ale(d: pd.DataFrame, bnd: dict, lev: dict, groups: dict[str, list[str]], tag: str, n: int = 60_000):
    """Devuelve (tabla resumen, dict de curvas). Guarda ale_{tag}.csv en E2.
    Criterio fijado antes de ver datos: |IQR| >= 1 pp (juicio, declarado). Ranking por magnitud, NO por IC."""
    E2.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for g, types in groups.items():
        base, Xs, par = build_base_tr(d, lev, types, n)
        pit = base.pitcher_anon_id.to_numpy()
        print(g, base.AutoPitchType.astype(str).value_counts(normalize=True).round(3).to_dict(),
              "| pitchers:", base.pitcher_anon_id.nunique())
        for h in HEADS:
            for f in FEATS:
                e, c, l, u, bt = ale_curves_tr(bnd[h]["model"], Xs, base, par, pit, f)
                dbt = bt[:, -1] - bt[:, 0]
                lo_d, hi_d = np.percentile(dbt, [2.5, 97.5])
                dl = c[-1] - c[0]
                am, an = int(c.argmax()), int(c.argmin())
                rows.append(dict(grupo=g, cabezal=h, feature=f, p5=e[0], p95=e[-1],
                                 delta_p95_p5=dl, ic_lo=lo_d, ic_hi=hi_d,
                                 delta_p25_p75=iqr_effect(e, c, base[f].to_numpy(float)),
                                 optimo_interior=(0 < am < len(c) - 1) and (c.max() - max(c[0], c[-1])) >= 0.5,
                                 valle_interior=(0 < an < len(c) - 1) and (min(c[0], c[-1]) - c.min()) >= 0.5))
                curves[(g, h, f)] = (e, c, l, u)
    tab = pd.DataFrame(rows)
    tab.to_csv(E2 / f"ale_{tag}.csv", index=False)
    return tab, curves


def plot_curves(curves: dict, groups, tag: str, show: bool = True) -> None:
    E2.mkdir(parents=True, exist_ok=True)
    for g in groups:
        for h, hname in HEADS.items():
            fig, axs = plt.subplots(2, 4, figsize=(17, 7))
            axs = axs.ravel()
            for ax, f in zip(axs, FEATS):
                e, c, l, u = curves[(g, h, f)]
                ax.plot(e, c)
                ax.fill_between(e, l, u, alpha=0.25)
                ax.axhline(0, color="0.7")
                ax.set_title(f + (" (origen)" if f in ("InducedVertBreak", "HB_arm") else ""))
                ax.set_ylabel("efecto (pp)")
            axs[-1].axis("off")
            fig.suptitle(f"{g} | {hname} | forma de origen transportada a Harp Helú | "
                         f"ALE p5-p95 (IC: solo muestreo de pitchers)")
            fig.tight_layout()
            fig.savefig(E2 / f"v3_{tag}_{g}_{h}.png", dpi=150)
            plt.show() if show else plt.close(fig)


# ----------------------------------------------------------------------------
# CELDAS DE NOTEBOOK (copiar una por una; no nombrar `d` a variables temporales)
# ----------------------------------------------------------------------------
# import importlib, stuffplus_eval as ev; importlib.reload(ev)
# d = ...  # frame_input.parquet + sr.build_frame(...) (fuentes oficiales: ver handoff 7 oct, sección 7)
# ev.verify_sources(d)
#
# # Pendiente 1 (tercil IVB, Four-Seam)
# t1 = ev.tercile_level_fourseam(d); print(t1.round(3).to_string())
#
# # Pendiente 2 (Changeup, un solo intento)
# t2, sb = ev.changeup_h2(d); print(t2.round(3).to_string())
#
# # Entregable 2 (ALE transportado)
# bnd, lev = ev.load_bundles(); ev.check_bundles(d, bnd, lev)
# tab_f, cur_f = ev.run_ale(d, bnd, lev, ev.FAM_TYPES, "familia_transportado")
# tab_t, cur_t = ev.run_ale(d, bnd, lev, ev.TYPES, "por_tipo")
# print(tab_t.pivot_table(index=["cabezal", "feature"], columns="grupo", values="delta_p25_p75").round(2))
# ev.plot_curves(cur_t, list(ev.TYPES), "tipo")
