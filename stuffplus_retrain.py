"""stuffplus_retrain.py — Reentrenamiento completo (6 oct 2026).
 
Pensado para notebook (celdas al final del archivo), sin argparse.
Principios anti-fuga:
  * Stuff+ = solo física medida en release (sin SpinAxis, sin ángulos, sin plato).
  * Early stopping SIEMPRE en un split interno por pitcher dentro del train,
    nunca en el fold de prueba.
  * Baselines (prevalencia y por conteo) calculados solo con el train de cada fold.
  * Categorías fijas (solo codificación), hiperparámetros congelados a priori.
  * Un checkpoint por (esquema, capa, target): si falla, no se pierde nada.
"""
from __future__ import annotations
 
import time
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import warnings
 
warnings.filterwarnings("ignore", message=".*eval_set.*")   # aviso de deprecación de LightGBM, inocuo
 
SEED = 42
OUT = Path("resultados_stuffplus/retrain_oct2026")
(OUT / "oof").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)
 
# ----------------------------------------------------------------------------
# 1. CAPAS DE FEATURES  (cada lista es contrato: nada fuera de aquí entra)
# ----------------------------------------------------------------------------
STUFF_NUM = ["RelSpeed", "SpinRate", "InducedVertBreak", "HB_arm",
             "RelHeight", "RelSide_arm", "Extension"]
STUFF_CAT = ["AutoPitchType", "PitcherThrows", "altitude_category"]
 
LOC_NUM = ["PlateLocHeight", "PlateLocSide_bat", "Balls", "Strikes"]
LOC_CAT = ["AutoPitchType", "PitcherThrows", "BatterSide", "altitude_category"]
 
TUNNEL_NUM = ["decision_separation_ft", "decision_separation_vs_prev_pitch_ft",
              "is_first_pitch_pa"]
ARSENAL_NUM = ["relspeed_diff_vs_primary", "inducedvertbreak_diff_vs_primary",
               "hb_diff_vs_primary_arm", "release_gap_vs_primary"]
ANGLE_NUM = ["VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle"]
 
_P_NUM = STUFF_NUM + LOC_NUM + ARSENAL_NUM + TUNNEL_NUM
_P_CAT = ["AutoPitchType", "PitcherThrows", "BatterSide", "altitude_category", "prev_pitch_type"]
 
LAYERS: dict[str, tuple[list[str], list[str]]] = {
    "stuff": (STUFF_NUM, STUFF_CAT),
    "location": (LOC_NUM, LOC_CAT),
    "pitching": (_P_NUM, _P_CAT),                                  # modelo principal / diseño
    # variantes SOLO para ablación (no se usan para recomendar arsenales):
    "pitching_angles": (_P_NUM + ANGLE_NUM, _P_CAT),               # ángulos codifican ubicación
    "pitching_usage": (_P_NUM + ["pitch_usage_pct"], _P_CAT),      # endógeno al pitcher
    "pitching_notunnel": ([c for c in _P_NUM if c not in TUNNEL_NUM], _P_CAT),
    "stuff_axis": (STUFF_NUM + ["spin_axis_sin", "spin_axis_cos"], STUFF_CAT),
}
 
# Columnas prohibidas en Stuff+ (plato, outcomes, redundantes/circulares)
FORBIDDEN_STUFF = {
    "PlateLocHeight", "PlateLocSide", "VertApprAngle", "HorzApprAngle", "VertRelAngle",
    "HorzRelAngle", "ZoneTime", "ZoneSpeed", "SpeedDrop", "EffectiveVelo", "SpinAxis",
    "Tilt", "pfxx", "pfxz", "ExitSpeed", "Angle", "Direction", "Distance", "Balls",
    "Strikes", "BatterSide",
}
 
 
def assert_no_leak_in_layers() -> None:
    num, cat = LAYERS["stuff"]
    bad = FORBIDDEN_STUFF.intersection(num + cat)
    assert not bad, f"Stuff+ contiene columnas prohibidas: {bad}"
    for name in ("location",):
        n, c = LAYERS[name]
        assert not ({"ExitSpeed", "is_whiff", "rv_bat"} & set(n + c))
 
 
# ----------------------------------------------------------------------------
# 2. FEATURES DERIVADAS
# ----------------------------------------------------------------------------
def add_derived(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Espeja por mano (signos medidos EMPÍRICAMENTE, no asumidos), crea prev_pitch_type
    y flags. Requiere que el orden de filas sea cronológico (validado: 0.34% violaciones).
    IMPORTANTE: se calcula ANTES de borrar filas, para no romper adyacencia dentro de la PA."""
    d = df.copy().reset_index(drop=True)
 
    sinker = d["AutoPitchType"].eq("Sinker")
    sgn_hb = d.loc[sinker].groupby("PitcherThrows")["HorzBreak"].median().apply(np.sign)
    sgn_rs = d.groupby("PitcherThrows")["RelSide"].median().apply(np.sign)
    if verbose:
        print("signo arm-side HB (mediana Sinker) por mano:", sgn_hb.to_dict())
        print("signo arm-side RelSide por mano:", sgn_rs.to_dict())
    assert sgn_hb.nunique() == 2 and sgn_rs.nunique() == 2, "No hay espejo claro por mano; revisar."
    print("PitcherThrows:", d["PitcherThrows"].value_counts(dropna=False).to_dict())
    print("BatterSide:", d["BatterSide"].value_counts(dropna=False).to_dict())
    # Solo Left/Right se espejan. 'Undefined' u otros -> NaN (no se adivina el lado del brazo).
    lr = ["Left", "Right"]
    hb_sign = d["PitcherThrows"].map({k: sgn_hb[k] for k in lr})
    d["HB_arm"] = d["HorzBreak"] * hb_sign
    d["RelSide_arm"] = d["RelSide"] * d["PitcherThrows"].map({k: sgn_rs[k] for k in lr})
    if "horzbreak_diff_vs_primary" in d:
        d["hb_diff_vs_primary_arm"] = d["horzbreak_diff_vs_primary"] * hb_sign
 
    bat_sign = d["BatterSide"].astype(str).str.upper().str[0].map({"R": 1.0, "L": -1.0})  # otro -> NaN
    d["PlateLocSide_bat"] = d["PlateLocSide"] * bat_sign
 
    if "spin_axis_sin" not in d and "SpinAxis" in d:        # solo para la ablación 'stuff_axis'
        a = np.deg2rad(pd.to_numeric(d["SpinAxis"], errors="coerce"))
        d["spin_axis_sin"], d["spin_axis_cos"] = np.sin(a), np.cos(a)
 
    assert "pa_id" in d, "Falta pa_id (construcción de PAs de la sesión 18 sept)."
    if "prev_pitch_type" not in d:
        d["prev_pitch_type"] = d.groupby("pa_id", sort=False)["AutoPitchType"].shift(1)
    first = d["prev_pitch_type"].isna().astype(float)
    first[d["pa_id"].isna()] = np.nan                       # sin PA válida: desconocido, no "primero"
    d["is_first_pitch_pa"] = first
    # Primer pitch de la PA: separación vs. previo = NaN (LightGBM lo maneja nativo; NO imputar mediana)
    return d
 
 
REQUIRED_RAW = [
    "PitchUID", "year", "game_anon_id", "pitcher_anon_id", "pa_id", "AutoPitchType",
    "PitcherThrows", "BatterSide", "altitude_category", "RelSpeed", "SpinRate", "SpinAxis",
    "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide", "Extension",
    "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle",
    "PlateLocHeight", "PlateLocSide", "Balls", "Strikes", "outside_strike_zone",
    "relspeed_diff_vs_primary", "inducedvertbreak_diff_vs_primary",
    "horzbreak_diff_vs_primary", "release_gap_vs_primary", "pitch_usage_pct",
    "decision_separation_ft", "decision_separation_vs_prev_pitch_ft",
    "is_swing", "is_whiff", "is_called_strike", "is_ball_in_play", "ExitSpeed", "hit_type",
]
 
 
def check_inputs(raw: pd.DataFrame) -> None:
    """Falla con un mensaje claro si el DataFrame no trae todo lo necesario."""
    missing = [c for c in REQUIRED_RAW if c not in raw.columns]
    assert not missing, f"Faltan columnas en el DataFrame de entrada: {missing}"
    print(f"OK: {len(raw):,} filas, todas las columnas requeridas presentes.")
 
 
def build_frame(raw: pd.DataFrame, rv_path: str | Path) -> pd.DataFrame:
    check_inputs(raw)
    d = add_derived(raw)
    rv = pd.read_parquet(rv_path)[["PitchUID", "rv_bat"]]
    d = d.merge(rv, on="PitchUID", how="left")
    n0 = len(d)
    n_alt = int(d["altitude_category"].isna().sum())
    n_pit = int(d["pitcher_anon_id"].isna().sum())
    # Se excluyen (no se imputan): sin altitud (~0.5%) y sin pitcher (no se puede agrupar por pitcher).
    # Se hace DESPUÉS de add_derived, así prev_pitch_type ya se calculó con las filas completas.
    keep = d["altitude_category"].notna() & d["pitcher_anon_id"].notna()
    d = d[keep].reset_index(drop=True)
    print(f"filas: {n0:,} -> {len(d):,} (sin altitud: {n_alt:,}; sin pitcher_anon_id: {n_pit:,})")
    return d
 
 
# ----------------------------------------------------------------------------
# 3. TARGETS (condicionales: separan "quién hace swing" de "qué pasa dado el swing")
# ----------------------------------------------------------------------------
def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce")
 
 
def make_targets(d: pd.DataFrame) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    swing, whiff, cs = _num(d, "is_swing"), _num(d, "is_whiff"), _num(d, "is_called_strike")
    bip = _num(d, "is_ball_in_play").eq(1)
    oz = _num(d, "outside_strike_zone").eq(1)
    ev = _num(d, "ExitSpeed")
    ht = d["hit_type"].astype("string").str.lower()
    rv = _num(d, "rv_bat")
    T = {
        "swing":       ("clf", swing, swing.notna()),
        "whiff_swing": ("clf", whiff, swing.eq(1) & whiff.notna()),
        "cs_take":     ("clf", cs, swing.eq(0) & cs.notna()),
        "chase_oz":    ("clf", swing, oz & swing.notna()),
        "hard_bip":    ("clf", (ev >= 95).astype(float), bip & ev.notna()),   # 95 mph = def. Statcast/MLB
        "gb_bip":      ("clf", ht.str.contains("ground", na=False).astype(float), bip & ht.notna()),
        "rv":          ("reg", rv, rv.notna()),                               # OFICIAL: rv_bat v3
    }
    return {k: (kind, y.to_numpy(float), np.asarray(m.fillna(False), dtype=bool))
            for k, (kind, y, m) in T.items()}
 
 
# ----------------------------------------------------------------------------
# 4. ESQUEMAS DE VALIDACIÓN  -> lista de (fold_id, idx_train, idx_test) posicionales
# ----------------------------------------------------------------------------
def folds_pitcher(d, n=5):
    g = d["pitcher_anon_id"].to_numpy()
    return [(i, tr, te) for i, (tr, te) in enumerate(GroupKFold(n).split(np.zeros(len(d)), groups=g))]
 
 
def folds_season_loso(d):
    yr = d["year"].to_numpy()
    return [(int(y), np.where(yr != y)[0], np.where(yr == y)[0]) for y in sorted(np.unique(yr))]
 
 
def folds_season_pitcher_strict(d, test_frac=0.3):
    """Temporada Y pitchers no vistos a la vez: test = 30% de pitchers de la temporada y;
    train = otras temporadas SIN esos pitchers (en ninguna temporada)."""
    rng = np.random.default_rng(SEED)
    yr, pit = d["year"].to_numpy(), d["pitcher_anon_id"].to_numpy()
    out = []
    for y in sorted(np.unique(yr)):
        ps = np.unique(pit[yr == y])
        held = set(rng.choice(ps, size=int(len(ps) * test_frac), replace=False))
        is_held = np.fromiter((p in held for p in pit), bool, count=len(pit))
        out.append((int(y), np.where((yr != y) & ~is_held)[0], np.where((yr == y) & is_held)[0]))
    return out
 
 
def folds_altitude(d, n=5):
    """Pitcher-disjunto Y altitud-disjunta: train No+Medium (pitchers fuera del fold),
    test Extreme (pitchers del fold). Se entrena SIN altitude_category (ver drop_alt)."""
    alt = d["altitude_category"].to_numpy()
    out = []
    for i, tr, te in folds_pitcher(d, n):
        out.append((i, tr[np.isin(alt[tr], ["No Altitude", "Medium Altitude"])],
                    te[alt[te] == "Extreme Altitude"]))
    return out
 
 
def folds_park(d, park_map: pd.DataFrame, fold_col="park_fold", ok_col="label_ok"):
    """park_map: una fila por juego con `game_anon_id`, fold_col (0..5, NaN para fragmentos)
    y label_ok. Ajusta nombres a como quedó park_map_final.parquet."""
    assert {"game_anon_id", fold_col, ok_col} <= set(park_map.columns), park_map.columns.tolist()
    m = d[["game_anon_id"]].merge(park_map[["game_anon_id", fold_col, ok_col]], on="game_anon_id", how="left")
    f = m[fold_col].to_numpy(float)
    ok = m[ok_col].fillna(False).to_numpy(bool)
    out = []
    for k in sorted(np.unique(f[~np.isnan(f)])):
        te = np.where((f == k) & ok)[0]
        tr = np.where((f != k) & ok & ~np.isnan(f))[0]
        out.append((int(k), tr, te))
    return out
 
 
SCHEMES = {
    "pitcher": (folds_pitcher, False),
    "season": (folds_season_loso, False),
    "season_pitcher": (folds_season_pitcher_strict, False),
    "altitude": (folds_altitude, True),      # True = quitar altitude_category de las features
    # "park": se construye aparte con folds_park(d, park_map)
}
 
 
# ----------------------------------------------------------------------------
# 5. ENTRENAMIENTO
# ----------------------------------------------------------------------------
PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
              max_bin=127, random_state=SEED, n_jobs=-1, verbosity=-1)   # congelados a priori
 
 
def _cat_levels(d, cats):
    return {c: sorted(d[c].dropna().astype(str).unique()) for c in cats}
 
 
def _design(d, layer, drop_alt, levels):
    num, cat = LAYERS[layer]
    if drop_alt:
        cat = [c for c in cat if c != "altitude_category"]
    X = d[num + cat].copy()
    for c in cat:
        X[c] = pd.Categorical(X[c].astype("object").where(X[c].notna(), None).astype("string"),
                              categories=levels[c])
    return X
 
 
def _fit(Xtr, ytr, gtr, kind):
    a, b = next(GroupShuffleSplit(1, test_size=0.1, random_state=SEED).split(Xtr, groups=gtr))
    m = LGBMClassifier(**PARAMS) if kind == "clf" else LGBMRegressor(**PARAMS)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])],
          callbacks=[early_stopping(50, verbose=False)])
    return m
 
 
def _predict(m, X, kind):
    return m.predict_proba(X)[:, 1] if kind == "clf" else m.predict(X)
 
 
def run_scheme(d, scheme, layers, targets, folds=None, force=False):
    T = make_targets(d)
    fn, drop_alt = SCHEMES.get(scheme, (None, False))
    folds = folds if folds is not None else fn(d)
    cats_all = sorted({c for l in layers for c in LAYERS[l][1]})
    levels = _cat_levels(d, cats_all)
    key = (d["Balls"].astype(int) * 10 + d["Strikes"].astype(int)).to_numpy()
    groups = d["pitcher_anon_id"].to_numpy()
    for layer in layers:
        X = _design(d, layer, drop_alt, levels)
        for t in targets:
            path = OUT / "oof" / f"{scheme}__{layer}__{t}.parquet"
            if path.exists() and not force:
                print("skip", path.name); continue
            kind, y, mask = T[t]
            t0, parts = time.time(), []
            for fid, tr, te in folds:
                tr, te = tr[mask[tr]], te[mask[te]]
                if len(tr) < 1000 or len(te) < 100:
                    print(f"  fold {fid}: muestra insuficiente, omitido"); continue
                m = _fit(X.iloc[tr], y[tr], groups[tr], kind)
                p = _predict(m, X.iloc[te], kind)
                base_prev = y[tr].mean()
                cm = pd.Series(y[tr]).groupby(key[tr]).mean()          # baseline por conteo, SOLO train
                base_cnt = pd.Series(key[te]).map(cm).fillna(base_prev).to_numpy()
                parts.append(pd.DataFrame({"PitchUID": d["PitchUID"].to_numpy()[te], "fold": fid,
                                           "y": y[te], "p": p, "base_prev": base_prev,
                                           "base_count": base_cnt, "pitcher_anon_id": groups[te],
                                           "year": d["year"].to_numpy()[te]}))
            if parts:
                pd.concat(parts).to_parquet(path, index=False)
            print(f"[{scheme}|{layer}|{t}] {time.time()-t0:.0f}s")
 
 
def fit_final(d, layer, target, drop_alt=False):
    """Modelo final sobre TODO el dato (para scores y simulador). Se guarda con joblib."""
    T = make_targets(d)
    kind, y, mask = T[target]
    levels = _cat_levels(d, LAYERS[layer][1])
    X = _design(d, layer, drop_alt, levels)
    idx = np.where(mask)[0]
    m = _fit(X.iloc[idx], y[idx], d["pitcher_anon_id"].to_numpy()[idx], kind)
    joblib.dump({"model": m, "layer": layer, "target": target, "levels": levels,
                 "features": LAYERS[layer]}, OUT / "models" / f"final__{layer}__{target}.joblib")
    return m
 
 
# ----------------------------------------------------------------------------
# 6. MÉTRICAS CON IC (bootstrap por PITCHER sobre OOF ya guardado: barato)
# ----------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
 
 
def _calib(y, p):
    slope = LogisticRegression(C=1e6, max_iter=200).fit(_logit(p).reshape(-1, 1), y).coef_[0, 0]
    bins = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b", observed=True).agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return slope, float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())
 
 
def _point(o, kind):
    y, p, bp, bc = o["y"], o["p"], o["base_prev"], o["base_count"]
    if kind == "clf":
        auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan
        b, b0, bcn = np.mean((p - y) ** 2), np.mean((bp - y) ** 2), np.mean((bc - y) ** 2)
        return {"auc": auc, "brier": b, "bss_prev": 1 - b / b0, "bss_count": 1 - b / bcn}
    rm, rb = np.sqrt(np.mean((y - p) ** 2)), np.sqrt(np.mean((y - bc) ** 2))
    return {"rmse": rm, "rmse_base_count": rb, "gain_pct": 100 * (1 - rm / rb)}
 
 
def summarize(scheme, layer, target, kind, B=200):
    o = pd.read_parquet(OUT / "oof" / f"{scheme}__{layer}__{target}.parquet")
    arr = {c: o[c].to_numpy() for c in ["y", "p", "base_prev", "base_count"]}
    row = {"scheme": scheme, "layer": layer, "target": target, "n": len(o), **_point(arr, kind)}
    if kind == "clf":
        row["calib_slope"], row["ece"] = _calib(arr["y"], arr["p"])
    idx = o.groupby("pitcher_anon_id").indices
    keys, rng = list(idx), np.random.default_rng(SEED)
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        ii = np.concatenate([idx[keys[j]] for j in pick])
        boots.append(_point({k: v[ii] for k, v in arr.items()}, kind))
    bdf = pd.DataFrame(boots)
    for c in bdf:
        row[f"{c}_lo"], row[f"{c}_hi"] = bdf[c].quantile([0.025, 0.975])
    return row
 
 
def summarize_all(schemes, layers, targets, B=200):
    T_kind = {"rv": "reg"}
    rows = []
    for s in schemes:
        for l in layers:
            for t in targets:
                if (OUT / "oof" / f"{s}__{l}__{t}.parquet").exists():
                    rows.append(summarize(s, l, t, T_kind.get(t, "clf"), B))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_summary.csv", index=False)
    return df
 
 
def paired_delta(scheme, layer_a, layer_b, target, B=200, kind="clf"):
    """Delta pareado (a - b) en AUC (clf) o RMSE (reg) con IC por bootstrap de pitchers.
    Usar para ablaciones: la diferencia pareada tiene IC mucho más angosto que comparar dos ICs."""
    a = pd.read_parquet(OUT / "oof" / f"{scheme}__{layer_a}__{target}.parquet").set_index("PitchUID")
    b = pd.read_parquet(OUT / "oof" / f"{scheme}__{layer_b}__{target}.parquet").set_index("PitchUID")
    j = a[["y", "p", "pitcher_anon_id"]].join(b[["p"]], rsuffix="_b", how="inner")
    f = (lambda y, p: roc_auc_score(y, p)) if kind == "clf" else (lambda y, p: -np.sqrt(np.mean((y - p) ** 2)))
    y, pa, pb = j["y"].to_numpy(), j["p"].to_numpy(), j["p_b"].to_numpy()
    point = f(y, pa) - f(y, pb)
    idx = j.reset_index().groupby("pitcher_anon_id").indices
    keys, rng = list(idx), np.random.default_rng(SEED)
    d = []
    for _ in range(B):
        ii = np.concatenate([idx[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        d.append(f(y[ii], pa[ii]) - f(y[ii], pb[ii]))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"a": layer_a, "b": layer_b, "target": target, "delta": point, "lo": lo, "hi": hi}
 
 
# ----------------------------------------------------------------------------
# 7. PUERTA ANTI-FUGA (reejecutable) y SCORE OFICIAL
# ----------------------------------------------------------------------------
def leak_probe(d, layer="stuff", frac=0.25, max_r2=0.30):
    """¿Las features de la capa predicen la ubicación en plato? Si R² > max_r2, hay fuga."""
    assert_no_leak_in_layers()
    levels = _cat_levels(d, LAYERS[layer][1])
    X = _design(d, layer, False, levels)
    ps = d["pitcher_anon_id"].drop_duplicates().sample(frac=frac, random_state=SEED)
    idx = np.where(d["pitcher_anon_id"].isin(ps).to_numpy())[0]
    res = {}
    for tgt in ("PlateLocHeight", "PlateLocSide_bat"):
        yy = d[tgt].to_numpy(float)[idx]
        pred = np.full(len(idx), np.nan)
        for tr, te in GroupKFold(3).split(idx, groups=d["pitcher_anon_id"].to_numpy()[idx]):
            m = LGBMRegressor(n_estimators=300, learning_rate=0.08, random_state=SEED, n_jobs=-1, verbosity=-1)
            m.fit(X.iloc[idx[tr]], yy[tr]); pred[te] = m.predict(X.iloc[idx[te]])
        ok = ~np.isnan(yy)
        res[tgt] = 1 - np.sum((yy[ok] - pred[ok]) ** 2) / np.sum((yy[ok] - yy[ok].mean()) ** 2)
    print(layer, res)
    return res
 
 
def official_score(scheme="pitcher", layer="pitching"):
    """Pitching+ oficial = 100 + 10*z( -(xRV_pred - E[rv|conteo]) ). Positivo = mejor para el pitcher."""
    o = pd.read_parquet(OUT / "oof" / f"{scheme}__{layer}__rv.parquet")
    raw = -(o["p"] - o["base_count"])
    o["pitching_plus"] = 100 + 10 * (raw - raw.mean()) / raw.std(ddof=0)
    return o
 
 
# ----------------------------------------------------------------------------
# CELDAS DE NOTEBOOK (copiar una por una)
# ----------------------------------------------------------------------------
# %% Celda A — cargar y congelar el marco de trabajo
# raw = pd.read_parquet("<data_seq con tunneling, pa_id, targets>")   # el data_pitch del equipo
# d = build_frame(raw, "resultados_stuffplus/eval_05oct2026/targets_runvalue_v3.parquet")
#
# %% Celda B — puerta anti-fuga (debe dar R² < 0.30 en 'stuff')
# for l in ["stuff", "location", "pitching"]: leak_probe(d, l)   # location/pitching SÍ pueden dar alto: tienen plato
#
# %% Celda C — humo (5% de pitchers) ANTES de la corrida larga: mide tiempos y detecta bugs
# ps = d.pitcher_anon_id.drop_duplicates().sample(frac=0.05, random_state=SEED)
# ds = d[d.pitcher_anon_id.isin(ps)].reset_index(drop=True)
# run_scheme(ds, "pitcher", ["stuff", "location", "pitching"], ["whiff_swing", "rv"], force=True)
#
# %% Celda D — corrida principal (orden de prioridad)
# TARGETS = ["rv", "whiff_swing", "swing", "chase_oz", "cs_take", "hard_bip", "gb_bip"]
# run_scheme(d, "pitcher", ["stuff", "location", "pitching"], TARGETS)
# run_scheme(d, "season", ["stuff", "location", "pitching"], TARGETS)
# run_scheme(d, "altitude", ["stuff", "pitching"], ["rv", "whiff_swing"])
# park = pd.read_parquet("resultados_stuffplus/eval_05oct2026/park_map_final.parquet")
# run_scheme(d, "park", ["stuff", "pitching"], ["rv", "whiff_swing"], folds=folds_park(d, park))
# run_scheme(d, "season_pitcher", ["pitching"], ["rv"])
#
# %% Celda E — ablaciones (solo esquema pitcher, pareadas)
# for b in ["pitching_angles", "pitching_usage", "pitching_notunnel"]:
#     run_scheme(d, "pitcher", [b], ["rv", "whiff_swing"])
# run_scheme(d, "pitcher", ["stuff_axis"], ["whiff_swing", "rv"])
# print(paired_delta("pitcher", "pitching", "location", "whiff_swing"))   # valor incremental de la física
# print(paired_delta("pitcher", "pitching", "pitching_notunnel", "rv", kind="reg"))
#
# %% Celda F — métricas con IC
# summarize_all(["pitcher","season","altitude","park","season_pitcher"],
#               ["stuff","location","pitching"], TARGETS, B=200)
