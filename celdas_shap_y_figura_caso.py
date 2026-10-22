# celdas_shap_y_figura_caso.py — 13 oct 2026. Formato de notebook ("# %%" = celda). NO corrido (sin datos en este entorno).
# Reglas de proceso: no nombrar `d` a temporales; no recargar `sr`; verificar fuentes antes (ev.verify_sources(d)).
#
# CORRECCIÓN DURA #2 — SHAP resumen del cabezal de Stuff+ (el brief llama a SHAP "herramienta esencial").
# Sin instalar `shap`: LightGBM calcula valores SHAP exactos con predict(pred_contrib=True).
# CRITERIO FIJADO ANTES: el ranking de |SHAP| medio debe coincidir en lo grueso con el de ALE
# (whiff: IVB y velocidad arriba; spin abajo). Si NO coincide, decláralo y explica (ALE vs SHAP
# responden preguntas distintas), no lo escondas. Describe al MODELO, no al béisbol.
 
# %% S1 — cargar modelo final y muestra
import joblib, numpy as np, pandas as pd
import stuffplus_retrain as sr
 
def shap_summary(target, layer="stuff", n=50_000, seed=42):
    pack = joblib.load(sr.OUT / "models" / f"final__{layer}__{target}.joblib")
    model, levels = pack["model"], pack["levels"]
    idx = d.sample(n=min(n, len(d)), random_state=seed).index
    X = sr._design(d.loc[idx].reset_index(drop=True), layer, False, levels)
    C = model.predict(X, pred_contrib=True)           # última columna = valor base
    C = np.asarray(C)[:, :-1]
    out = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(C).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    out["pct_del_total"] = 100 * out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    out["target"] = target
    return out
 
tabSHAP = pd.concat([shap_summary(t) for t in ["whiff_swing", "gb_bip"]])
print(tabSHAP.round(4).to_string(index=False))
tabSHAP.to_csv(sr.OUT / "shap_stuff_resumen.csv", index=False)
 
# Lectura para el documento (una frase, sin sobre-vender):
# "Como contraste con ALE, el |SHAP| medio del cabezal de Stuff+ ordena las variables así: ... .
#  Usamos ALE para las curvas porque las variables físicas están correlacionadas (IVB, velocidad, spin),
#  y ALE no evalúa combinaciones fuera de los datos; SHAP resume la contribución al modelo."
# Nota: con variables correlacionadas el SHAP reparte crédito entre ellas; no lo narres como efecto causal.
 
# %% F1 — CORRECCIÓN #4: mapa release -> decisión -> plato de un pitcher de caso (pitcher_00841, Splitter)
# ADVERTENCIA CRÍTICA: NO uses plot_arsenal.py ni features.trajectory_landmarks del repo original.
# Tienen los dos bugs ya corregidos (tiempo de vuelo con fallback lineal; eje x reflejado) y reloj propio
# por pitch. Usa stuffplus_core (normalize_types con x negado + build_primary_tunnel_features con reloj común).
import matplotlib.pyplot as plt
from stuffplus_core import normalize_types, build_primary_tunnel_features   # ajusta nombres si difieren
# raw = tu .pkl crudo de ISAC (necesita x0..az0). NO el frame_input.parquet sin esas columnas.
PIT, YR = "pitcher_00841", 2025
sub = normalize_types(raw.loc[(raw["pitcher_anon_id"] == PIT) & (raw["year"] == YR)].copy())
L = build_primary_tunnel_features(sub)
chk = L["decision_t_common_s"] < L["time_to_plate_s_own"]
print("decisión antes del plato en", round(100 * chk.mean(), 1), "% de filas (esperado 100)")
prof = pd.concat([sub[["AutoPitchType"]], L], axis=1).groupby("AutoPitchType", observed=True).mean(numeric_only=True)
prim = L["primary_pitch_type"].dropna().iloc[0]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (pt, ttl) in zip(ax, [("release", "Release"), ("decision", "Decisión (150 ms antes de que la recta llegue)"), ("plate", "Plato")]):
    for k, r in prof.iterrows():
        a.scatter(r[f"{pt}_x"], r[f"{pt}_z"], s=80, marker="s" if k == prim else "o")
        a.annotate(str(k), (r[f"{pt}_x"], r[f"{pt}_z"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    a.set_title(ttl, fontsize=10); a.set_xlabel("Lateral (ft)"); a.set_ylabel("Altura (ft)")
    a.set_aspect("equal", adjustable="datalim"); a.grid(alpha=.2)
# líneas punteadas decisión -> plato (diferido el 20 sept: "hacia el final")
for k, r in prof.iterrows():
    ax[2].plot([r["decision_x"], r["plate_x"]], [r["decision_z"], r["plate_z"]], ls=":", lw=1, color="0.4")
drop = prof.loc[prim, "release_z"] - prof.loc[prim, "plate_z"]
fig.suptitle(f"{PIT}, {YR}. La recta primaria ({prim}) cae {drop:.1f} ft del release al plato. Asociativo, retrospectivo.", fontsize=10)
fig.tight_layout(); fig.savefig("fig_caso_release_decision_plato.png", dpi=200)
# Chequeo antes de publicar: plate_z debe estar cerca de la zona (≈1.5–3.5 ft) y plate_y ≈ 0 con el fix de tiempo.
# Si el pitcher no tiene ≥100 pitches de la recta primaria, cambia PIT/YR (usa uno con celdas calificadas).
 
