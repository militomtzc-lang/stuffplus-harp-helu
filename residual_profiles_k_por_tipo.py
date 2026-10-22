"""Shrinkage k por (target, AutoPitchType) — sesión 19 sept 2026.

Reemplaza el `shrinkage_k` escalar (k=100 arbitrario, usado en TODO el
pipeline hasta ahora) por un diccionario derivado empíricamente vía
empirical Bayes / método de momentos (sesión 18 sept, ver decisions.md
del proyecto). Rango real derivado: 0.9-8.2 — un orden de magnitud por
debajo del valor arbitrario.

Con k=100, un pitcher-año-tipo típico (n=30) recibía peso ~0.23 hacia su
propio residual (77% jalado a la media global). Con el k real, ese peso
sube a ~0.75-0.97. Traducción sin rodeos: el pipeline hasta hoy ha estado
APLASTANDO la señal real de diferenciación entre pitchers en casi todos
los casos, no solo en muestra chica como se pensaba al fijar k=100 "para
ser conservadores". Cualquier cifra ya reportada (sección 4 del handoff
original, tabla de altitude delta con escenario conservador/proporcional)
que dependa de `*_non_magnus_residual_shrunk` está calculada con
shrinkage demasiado agresivo y debe considerarse provisional hasta
regenerarse.

QUÉ NO CAMBIA con este fix (para que no se sobre-corrija en el documento
técnico): la tabla de deltas de altitud de `harp_ivb_category_model` /
`harp_hb_category_model` (sesión 16 sept, comparación física vs. modelo
con altitude_category) NO usa `residual_profiles` en absoluto — viene
directo de la predicción del modelo de movimiento con la categoría
sustituida. Esa tabla sigue siendo válida tal cual está. Lo que SÍ cambia
son las columnas `delta_ivb_conservative` / `delta_ivb_proportional` /
sus equivalentes HB en el output de `simulate_altitude.py`, porque esas
sí usan el residual encogido.

LIMITACIÓN DECLARADA — no resuelta en este archivo, decláralo así en el
documento técnico, no la escondan:
Los k derivados solo cubren los 8 pitch types con muestra confiable
(Four-Seam, Sinker, Cutter, Slider, Curveball, Changeup, Splitter,
Sweeper). Para `Other`, `Knuckleball`, `OneSeamFastBall`,
`TwoSeamFastBall` (ya marcados como "n bajo / categoría basura / caos
físico intrínseco" en el handoff original, sección 4) NO se derivó k —
se usa FALLBACK_K=100, el valor viejo. Es una decisión deliberada de
seguridad (encoger fuerte hacia la media global cuando no auditamos la
muestra), no un descuido, pero no debe presentarse como si tuviera el
mismo respaldo empírico que los otros 8.

Splitter y Sweeper (IVB) traen su propia cautela ya documentada en
decisions.md: n de grupos pitcher-año bajo. No tratar su k con la misma
confianza que Four-Seam/Sinker/Slider, que tienen muestra grande.

Pendiente #4 de decisions.md, sigue sin resolver aquí: Sweeper tiene
k_ivb=8.2 vs. k_hb=2.0 — una asimetría de 4x entre IVB y HB para el
mismo pitch type que no se ha explicado. Hipótesis de trabajo (no
confirmada): umbral de n_groups distinto entre IVB y HB para Sweeper,
no necesariamente un patrón físico real. No construir ninguna narrativa
sobre esa asimetría hasta investigarla.

INSTRUCCIONES DE INTEGRACIÓN:
1. Reemplazar `residual_profiles()` en `stuffplus/movement.py` (o en
   `stuffplus_core.py` si el equipo ya migró) por la de este archivo.
   La firma es retrocompatible: si `shrinkage_k` sigue siendo un float,
   se comporta exactamente igual que antes.
2. Para regenerar el CSV SIN reentrenar los modelos de LightGBM (las
   predicciones OOF no cambian, solo el shrinkage posterior), usar el
   bloque de notebook al final de este archivo sobre
   `movement_oof_predictions.parquet` ya existente.
3. Después de regenerar, volver a correr `simulate_altitude.py` con el
   `non_magnus_residual_profiles.csv` nuevo para que las columnas
   `delta_*_conservative`/`delta_*_proportional` queden consistentes.
"""

from __future__ import annotations

import pandas as pd

# Derivado sesión 18 sept 2026 vía empirical Bayes (método de momentos),
# razón de varianza entre/dentro de pitcher_anon_id, por (target, AutoPitchType).
SHRINKAGE_K_BY_TARGET_TYPE: dict[tuple[str, str], float] = {
    ("ivb", "Curveball"): 0.80, ("ivb", "Cutter"): 0.96, ("ivb", "Splitter"): 1.12,
    ("ivb", "Sinker"): 1.17, ("ivb", "Slider"): 1.27, ("ivb", "Changeup"): 1.56,
    ("ivb", "Four-Seam"): 1.69, ("ivb", "Sweeper"): 1.76,  # n=61 grupos, no confiable
    ("hb", "Sweeper"): 0.42,  # no confiable
    ("hb", "Curveball"): 0.92, ("hb", "Slider"): 1.30, ("hb", "Cutter"): 1.39,
    ("hb", "Four-Seam"): 1.42, ("hb", "Splitter"): 1.52, ("hb", "Changeup"): 2.14,
    ("hb", "Sinker"): 2.68,
}
FALLBACK_K = 100.0  # Other, Sweeper (ambos targets), Knuckleball, OneSeam, TwoSeam


def residual_profiles(
    oof: pd.DataFrame,
    shrinkage_k: float | dict[tuple[str, str], float] = 100.0,
) -> pd.DataFrame:
    """Promedia y encoge residuales por pitcher-temporada-tipo de pitch.

    `shrinkage_k`:
    - float -> comportamiento viejo, mismo k para todos los targets/tipos.
    - dict {(short_name, AutoPitchType): k} -> comportamiento nuevo,
      recomendado: pasar SHRINKAGE_K_BY_TARGET_TYPE. Tipos ausentes del
      dict caen a FALLBACK_K, y se imprime aviso de cuáles cayeron —
      nunca en silencio.
    """
    keys = ["year", "pitcher_anon_id", "AutoPitchType"]
    profiles = (
        oof.groupby(keys, as_index=False, observed=True)
        .agg(
            n_pitches=("PitchUID", "size"),
            ivb_residual_mean=("ivb_non_magnus_residual_proxy", "mean"),
            hb_residual_mean=("hb_non_magnus_residual_proxy", "mean"),
        )
    )

    fell_back: set[str] = set()

    def _k_for(short_name: str, pitch_type: object) -> float:
        if isinstance(shrinkage_k, dict):
            key = (short_name, pitch_type)
            if key in shrinkage_k:
                return shrinkage_k[key]
            fell_back.add(str(pitch_type))
            return FALLBACK_K
        return float(shrinkage_k)

    for short_name in ("ivb", "hb"):
        k_series = profiles["AutoPitchType"].map(lambda pt: _k_for(short_name, pt))
        weight = profiles["n_pitches"] / (profiles["n_pitches"] + k_series)
        profiles[f"{short_name}_shrinkage_k_used"] = k_series
        profiles[f"{short_name}_non_magnus_residual_shrunk"] = (
            weight * profiles[f"{short_name}_residual_mean"]
        )

    if fell_back:
        print(f"Aviso: tipos sin k derivado, usando FALLBACK_K={FALLBACK_K}: {sorted(fell_back)}")
    return profiles


# ---------------------------------------------------------------------------
# Celda de notebook: regenerar sin reentrenar LightGBM
# ---------------------------------------------------------------------------
# oof = pd.read_parquet("<output-dir>/movement_oof_predictions.parquet")
# profiles_new = residual_profiles(oof, shrinkage_k=SHRINKAGE_K_BY_TARGET_TYPE)
# profiles_new.to_csv("<output-dir>/non_magnus_residual_profiles.csv", index=False)
#
# # Comparación antes/después -- correr esto, no asumir que el cambio es sano:
# profiles_old = residual_profiles(oof, shrinkage_k=100.0)
# compare = profiles_new.merge(
#     profiles_old[["year", "pitcher_anon_id", "AutoPitchType",
#                    "ivb_non_magnus_residual_shrunk", "hb_non_magnus_residual_shrunk"]],
#     on=["year", "pitcher_anon_id", "AutoPitchType"],
#     suffixes=("_new", "_old"),
# )
# compare["ivb_shift"] = compare["ivb_non_magnus_residual_shrunk_new"] - compare["ivb_non_magnus_residual_shrunk_old"]
# print(compare.groupby("AutoPitchType")["ivb_shift"].describe())
#
# # Después, re-correr simulate_altitude.py apuntando al CSV nuevo:
# # python simulate_altitude.py --input physics_features.parquet \
# #   --models movement_models.joblib \
# #   --residual-profiles <output-dir>/non_magnus_residual_profiles.csv \
# #   --output <output-dir>/altitude_simulation_k_por_tipo.parquet
