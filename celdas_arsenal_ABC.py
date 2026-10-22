# celdas_arsenal_ABC.py — 10 oct 2026. Formato de notebook (cada "# %%" es una celda).
# ESTADO: A1-A3 y C1 corren sin errores con datos sintéticos (solo detecta bugs de código; los números sintéticos no significan nada).
# La celda B depende de LightGBM y de stuffplus_retrain.py y NO se ejecutó.
#
# CRITERIOS FIJADOS ANTES DE VER RESULTADOS
#  B  (ablación de arsenal): aporte "medible" = IC95 del delta pareado excluye 0; aporte "relevante" =
#     |ΔAUC| >= 0.002 o Δrv >= 0.05 pp (criterio mío, declarado). La cadena debe sumar el total.
#  A  (contraste de velocidad): "sobrevive" solo si en S2 (con controles) y en S3 (mismo pitcher entre años)
#     el signo es el esperado en rv (negativo) y en whiff|swing (positivo) y el IC95 de pitchers excluye 0,
#     para Slider y Changeup. Si solo sobrevive el crudo (S0/S1), es confusión entre pitchers.
#  C1 (altitud y arsenal): "la altitud comprime la relación" si el IC95 del delta Extreme − No excluye 0.
#  Todo es asociativo. Los IC cubren solo variación entre pitchers.
#
# REGLAS DE PROCESO (7 oct): no nombrar d a variables temporales; no recargar sr (se pierde el parche de `bip`);
# verificar fuentes antes de empezar.
 
# %% Celda 0 — verificar el marco
import time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import stuffplus_retrain as sr, stuffplus_eval as ev
 
ev.verify_sources(d)                                         # fuentes oficiales contra OOF
assert len(d) == 629_026, len(d)
assert abs(d["rv_bat"].std() - 0.1848) < 0.004, f"rv_bat no parece v4: std={d['rv_bat'].std():.4f} (v3 ~ 0.2485)"
assert d["is_first_pitch_pa"].nunique() > 1, "is_first_pitch_pa sigue constante: aplicar el fix del 8 oct"
print("is_first_pitch_pa:", d["is_first_pitch_pa"].value_counts(dropna=False).to_dict())
 
# %% Celda B1 — capas nuevas (cadena anidada) y corrida
# location  ->  stuffloc (+física)  ->  stuffloc_diff (+diferenciales vs recta)  ->  pitching (+tunneling y pitch previo)
STUFF_NUM, LOC_NUM, ARS_NUM = sr.STUFF_NUM, sr.LOC_NUM, sr.ARSENAL_NUM
CAT_SL = ["AutoPitchType", "PitcherThrows", "BatterSide", "altitude_category"]
sr.LAYERS["stuffloc"]        = (STUFF_NUM + LOC_NUM, CAT_SL)
sr.LAYERS["stuffloc_diff"]   = (STUFF_NUM + LOC_NUM + ARS_NUM, CAT_SL)
sr.LAYERS["pitching_nodiff"] = ([c for c in sr._P_NUM if c not in ARS_NUM], sr._P_CAT)
for l in ["stuffloc", "stuffloc_diff", "pitching_nodiff"]:
    num, cat = sr.LAYERS[l]; miss = [c for c in num + cat if c not in d.columns]; assert not miss, (l, miss)
 
SMOKE = True            # True: 5% de pitchers para medir tiempos y detectar bugs. Luego False.
if SMOKE:
    ps = d["pitcher_anon_id"].drop_duplicates().sample(frac=0.05, random_state=sr.SEED)
    dd = d[d["pitcher_anon_id"].isin(ps)].reset_index(drop=True); SCHEME = "smoke_v2"
else:
    dd = d; SCHEME = "pitcher_v2"       # nombre nuevo: no pisa los OOF del 6-7 oct (que son previos al fix)
 
LAYERS_B = ["location", "stuffloc", "stuffloc_diff", "pitching", "pitching_nodiff"]
t0 = time.time()
sr.run_scheme(dd, SCHEME, LAYERS_B, ["whiff_swing", "rv"], folds=sr.folds_pitcher(dd))
print(f"{(time.time() - t0) / 60:.1f} min para whiff_swing y rv")
# Si el tiempo alcanza (con SMOKE=False):
# sr.run_scheme(dd, SCHEME, LAYERS_B, ["gb_bip", "chase_oz"], folds=sr.folds_pitcher(dd))
 
# %% Celda B2 — control y tabla de aportes
o = pd.read_parquet(sr.OUT / "oof" / f"{SCHEME}__pitching__whiff_swing.parquet")
print("AUC pitching whiff|swing:", round(roc_auc_score(o["y"], o["p"]), 4), "(con SMOKE=False esperado ~0.758)")
 
CHAIN = [("stuffloc", "location", "física, dada la ubicación"),
         ("stuffloc_diff", "stuffloc", "+ diferenciales vs recta"),
         ("pitching", "stuffloc_diff", "+ tunneling y pitch previo"),
         ("pitching", "location", "TOTAL Pitching+ - Location+"),
         ("pitching", "pitching_nodiff", "diferenciales, dado todo lo demás")]
 
def table_B(targets, B=200):
    rows = []
    for t in targets:
        kind = "reg" if t == "rv" else "clf"
        for a_, b_, lab in CHAIN:
            r = sr.paired_delta(SCHEME, a_, b_, t, B=B, kind=kind)
            if kind == "reg":                                   # RMSE -> pp de ganancia
                for k in ("delta", "lo", "hi"):
                    r[k] = 100 * r[k] / 0.1848
            r["pieza"], r["unidad"] = lab, ("pp rv" if kind == "reg" else "AUC")
            rows.append(r)
    return pd.DataFrame(rows)[["target", "pieza", "unidad", "delta", "lo", "hi"]]
 
tabB = table_B(["whiff_swing", "rv"])
print(tabB.round(4).to_string(index=False))
for t in tabB["target"].unique():                              # la cadena debe sumar el total (misma muestra)
    x = tabB[tabB["target"] == t].set_index("pieza")["delta"]
    print(t, "| suma de piezas:", round(x.iloc[:3].sum(), 4), "| total:", round(x["TOTAL Pitching+ - Location+"], 4))
 
# %% Celda A1 — insumos: xRV v4 centrado por conteo y recta primaria por pitcher-año
FB, SEC = ["Four-Seam", "Sinker"], ["Slider", "Changeup", "Curveball"]
MIN_N = 100                                                    # mínimo de pitches por celda (secundario y recta); declarado
a = d[["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "altitude_category", "RelSpeed", "Balls", "Strikes",
       "rv_bat", "is_swing", "is_whiff"]].copy()
a["AutoPitchType"] = a["AutoPitchType"].astype(str)
a["cnt"] = a["Balls"].astype(int) * 10 + a["Strikes"].astype(int)
a["rv_c"] = a["rv_bat"] - a.groupby("cnt")["rv_bat"].transform("mean")        # centrado por conteo, como Location+/Pitching+
_sw = pd.to_numeric(a["is_swing"], errors="coerce"); _wh = pd.to_numeric(a["is_whiff"], errors="coerce")
a["wh"] = _wh.where(_sw.eq(1))                                                # whiff dado swing
a["ext"] = (a["altitude_category"] == "Extreme Altitude").astype(float)
n_fb = a[a["AutoPitchType"].isin(FB)].groupby(["pitcher_anon_id", "year", "AutoPitchType"]).size().rename("n").reset_index()
prim = (n_fb.sort_values("n").groupby(["pitcher_anon_id", "year"]).tail(1)
        .rename(columns={"AutoPitchType": "prim"})[["pitcher_anon_id", "year", "prim"]])
a = a.merge(prim, on=["pitcher_anon_id", "year"], how="left")
a["is_prim"] = a["AutoPitchType"].eq(a["prim"])
print("pitcher-años con recta primaria:", len(prim), "| rv_c medio (debe ser ~0):", round(a["rv_c"].mean(), 6))
 
def build_cells(keys):
    fb = (a[a["is_prim"]].groupby(keys)
          .agg(fb_v=("RelSpeed", "mean"), fb_n=("RelSpeed", "size"), fb_rvc=("rv_c", "mean")).reset_index())
    s = (a[a["AutoPitchType"].isin(SEC)].groupby(keys + ["AutoPitchType"])
         .agg(n=("RelSpeed", "size"), sec_v=("RelSpeed", "mean"), rv=("rv_bat", "mean"), rv_c=("rv_c", "mean"),
              wh_sum=("wh", "sum"), sw_n=("wh", "count"), ext=("ext", "mean")).reset_index())
    c = s.merge(fb, on=keys, how="inner")
    c = c[(c["n"] >= MIN_N) & (c["fb_n"] >= MIN_N)].copy()
    c["gap"] = c["fb_v"] - c["sec_v"]                                          # mph: recta menos secundario
    c["rv100"], c["rvc100"], c["fb_rvc100"] = 100 * c["rv"], 100 * c["rv_c"], 100 * c["fb_rvc"]
    c["whiff_sw"] = 100 * c["wh_sum"] / c["sw_n"].where(c["sw_n"] >= 30)       # % whiff dado swing, mínimo 30 swings
    return c
 
lvl = build_cells(["pitcher_anon_id", "altitude_category"])
lvl = lvl[lvl["altitude_category"].isin(["No Altitude", "Extreme Altitude"])]
panel = build_cells(["pitcher_anon_id", "year"])
print("celdas pitcher-nivel:", lvl.groupby(["AutoPitchType", "altitude_category"]).size().to_dict())
print("celdas pitcher-año  :", panel.groupby("AutoPitchType").size().to_dict())
 
# %% Celda A2 — escalera de especificaciones (S0 crudo -> S2 con controles) a nivel pitcher, como el compañero
def _resid(v, Z):
    if Z is None:
        return v - v.mean()
    Z1 = np.column_stack([np.ones(len(v)), Z]); return v - Z1 @ np.linalg.lstsq(Z1, v, rcond=None)[0]
 
def _pcorr(df, x, y, ctrl):
    Z = df[ctrl].to_numpy(float) if ctrl else None
    return np.corrcoef(_resid(df[x].to_numpy(float), Z), _resid(df[y].to_numpy(float), Z))[0, 1]
 
def corr_boot(df, x, y, ctrl=(), B=400, seed=42):
    ctrl = list(ctrl); df = df.dropna(subset=[x, y] + ctrl).reset_index(drop=True)
    pt = _pcorr(df, x, y, ctrl)
    idx = df.groupby("pitcher_anon_id").indices; keys = list(idx); rng = np.random.default_rng(seed)
    bs = [_pcorr(df.iloc[np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))])], x, y, ctrl)
          for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi, len(df), len(keys)
 
COLEGA = {("Slider", "No Altitude"): -0.379, ("Slider", "Extreme Altitude"): -0.301,
          ("Changeup", "No Altitude"): -0.235, ("Changeup", "Extreme Altitude"): -0.253}   # r reportado con su xRV
STEPS = [("S0 crudo (xRV v4)", "rv100", []),
         ("S1 + centrado por conteo", "rvc100", []),
         ("S2 + vel. del secundario y calidad de la recta", "rvc100", ["sec_v", "fb_rvc100"]),
         ("S2 en whiff|swing (signo esperado: +)", "whiff_sw", ["sec_v", "fb_rvc100"])]
rows = []
for t in SEC:
    for alt in ["No Altitude", "Extreme Altitude"]:
        c = lvl[(lvl["AutoPitchType"] == t) & (lvl["altitude_category"] == alt)]
        if len(c) < 30: continue
        for lab, y, ctrl in STEPS:
            r, lo, hi, n, npit = corr_boot(c, "gap", y, ctrl)
            rows.append(dict(tipo=t, altitud=alt[:3], escalon=lab, r=r, lo=lo, hi=hi, n=n, r_colega=COLEGA.get((t, alt))))
tabA = pd.DataFrame(rows); print(tabA.round(3).to_string(index=False))
 
# %% Celda A3 — S3: el mismo pitcher entre años (efectos fijos de pitcher y de año)
def fe_slope(c, x, y, ctrl=(), B=400, seed=42):
    cols = [x, y] + list(ctrl)
    c = c.dropna(subset=cols).copy()
    c = c[c.groupby("pitcher_anon_id")[x].transform("size") >= 2].copy()       # solo pitchers con >= 2 años
    for k in cols:
        c[k] = c[k] - c.groupby("year")[k].transform("mean")                   # quita tendencia de la liga por año
        c[k] = c[k] - c.groupby("pitcher_anon_id")[k].transform("mean")        # efecto fijo de pitcher
    c = c.reset_index(drop=True)
    X = c[[x] + list(ctrl)].to_numpy(float); Y = c[y].to_numpy(float)
    slope = lambda ii: np.linalg.lstsq(X[ii], Y[ii], rcond=None)[0][0]
    idx = c.groupby("pitcher_anon_id").indices; keys = list(idx); rng = np.random.default_rng(seed)
    pt = slope(np.arange(len(c)))
    bs = [slope(np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))])) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi, len(c), len(keys)
 
rows = []
for t in SEC:
    c = panel[panel["AutoPitchType"] == t]
    for lab, y in [("S3 rv centrado (pendiente por mph de brecha)", "rvc100"), ("S3 whiff|swing (pp por mph)", "whiff_sw")]:
        if c.groupby("pitcher_anon_id").size().ge(2).sum() < 30:
            continue
        b, lo, hi, n, npit = fe_slope(c, "gap", y, ["ext"])                   # 'ext' = % de pitches en altitud extrema ese año
        rows.append(dict(tipo=t, escalon=lab, pendiente=b, lo=lo, hi=hi, n_celdas=n, n_pitchers=npit))
tabA3 = pd.DataFrame(rows); print(tabA3.round(3).to_string(index=False))
# Lectura: si S0/S1 son negativos (como el compañero) pero S2/S3 se acercan a 0 o cruzan 0, el contraste era confusión
# entre pitchers (recta más rápida = mejor pitcher). Si S3 se mantiene, es el resultado limpio.
 
# %% Celda C1 — ¿la altitud cambia las relaciones del arsenal? (mismo pitcher, Extreme - No)
VARS_C = {"relspeed_diff_vs_primary": "mph", "inducedvertbreak_diff_vs_primary": "pulg", "hb_diff_vs_primary_arm": "pulg",
          "release_gap_vs_primary": "unidad del dato", "decision_separation_ft": "ft",
          "decision_separation_vs_prev_pitch_ft": "ft"}
TYPES6 = ["Four-Seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup"]
 
def within_delta(df, var, group="AutoPitchType", minobs=30, minp=15, B=300, seed=42):
    g = df[["pitcher_anon_id", "altitude_category", group, var]].copy()
    g[var] = pd.to_numeric(g[var], errors="coerce"); g = g.dropna()
    g = g[g["altitude_category"].isin(["No Altitude", "Extreme Altitude"])]
    m = g.groupby(["pitcher_anon_id", group, "altitude_category"])[var].agg(["mean", "size"]).unstack("altitude_category")
    m.columns = [f"{s}_{alt[0]}" for s, alt in m.columns]                      # mean_E, mean_N, size_E, size_N
    m = m.dropna().reset_index()
    m = m[(m["size_E"] >= minobs) & (m["size_N"] >= minobs)].copy()
    m["dlt"] = m["mean_E"] - m["mean_N"]; m["w"] = m["size_E"] * m["size_N"] / (m["size_E"] + m["size_N"])
    out, rng = [], np.random.default_rng(seed)
    for k, h in m.groupby(group):
        if len(h) < minp: continue
        dl, w = h["dlt"].to_numpy(), h["w"].to_numpy()
        pt = np.average(dl, weights=w)
        bs = [np.average(dl[i], weights=w[i]) for i in (rng.integers(0, len(h), len(h)) for _ in range(B))]
        out.append(dict(grupo=k, delta=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                        n_pitchers=len(h), base_No=np.average(h["mean_N"], weights=w)))
    return pd.DataFrame(out)
 
dt = d[d["AutoPitchType"].astype(str).isin(TYPES6)].copy(); dt["AutoPitchType"] = dt["AutoPitchType"].astype(str)
tabC = pd.concat([within_delta(dt, v).assign(variable=v, unidad=u) for v, u in VARS_C.items()])
tabC = tabC[["variable", "unidad", "grupo", "delta", "lo", "hi", "n_pitchers", "base_No"]]
print(tabC.round(3).to_string(index=False))
 
# Lectura orientativa en chase: asociación del 7 oct (IQR de separación 0.759 ft <-> 9.0 pp de chase; lineal, no causal).
sep = tabC[tabC["variable"] == "decision_separation_ft"].copy()
sep["chase_implicito_pp"] = -(9.0 / 0.759) * sep["delta"]
print(sep[["grupo", "delta", "lo", "hi", "chase_implicito_pp"]].round(3).to_string(index=False))
# Nota: el efecto en chase NO se verifica aquí; es una extrapolación lineal. La verificación es la contrafactual C2
# (se escribe después de ver estos deltas; si todos cruzan 0, C2 no hace falta).
 
