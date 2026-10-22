"""Fix de tunneling — reloj común + separación x-z (sesión 16 sept 2026).

INSTRUCCIONES DE INTEGRACIÓN:
Este archivo NO es standalone para producción. Está pensado para pegarse
dentro de `stuffplus/features.py`, justo después de `build_arsenal_overlap`
(o donde el equipo prefiera). Reutiliza `_choose_primary_pitch` y
`_time_to_plate`, que YA EXISTEN en ese archivo — no las redefinas, bórralas
de aquí si las copias/pegas directo.

Corre solo sobre el .pkl crudo de ISAC (necesita x0..az0), igual que
`trajectory_landmarks` / `build_arsenal_overlap`, NO sobre
`physics_features.parquet` (no conserva esas columnas).

Contexto del bug que corrige (ver handoff_stuffplus.md sección 3 y
handoff_addendum_16sept2026.md):

1. `trajectory_landmarks` original calculaba `decision_t` con el tiempo de
   vuelo PROPIO de cada pitch (`plate_t_own - 0.150`). Cada tipo de pitch
   quedaba comparado en un instante de reloj distinto desde el release —
   curva y recta no se comparaban al mismo tiempo. Confirmado con datos
   reales: 43.9% de las celdas pitcher-año-tipo (n≥100) mostraban separación
   en "decisión" mayor que en el plato, patrón físicamente inconsistente con
   tunneling (las trayectorias deberían divergir hacia el plato, no al revés).

2. Primer intento de fix (reloj común anclado a la recta primaria) NO
   resolvió el problema por sí solo (43.4%, prácticamente igual). Causa real
   encontrada: la métrica de separación usaba distancia euclidiana 3D
   (x, y, z). A tiempo de reloj común, un pitch más lento (p.ej. curveball)
   está a una profundidad `y` mucho mayor del plato que la recta, simplemente
   por ser más lento — esa diferencia de profundidad no es lo que el
   bateador percibe como separación de trayectoria, e inflaba la separación
   en "decisión" de forma artificial para pitches lentos.

3. Fix final: (a) reloj común anclado a la MEDIANA del tiempo de vuelo de la
   recta primaria por pitcher-año, y (b) separación medida SOLO en el plano
   x-z (lateral + vertical — lo que el bateador ve), excluyendo `y`.
   Resultado validado con datos reales (pitcher_01067, 2024, y agregados
   n≥100): inversion_rate cae a 4-23% para la mayoría de tipos (patrón
   correcto), con Curveball como excepción a 38.8% — verificado que NO decae
   con el tamaño de muestra del ancla, se interpreta como hallazgo físico
   real (el curveball tunnelea mejor en el punto de decisión, rompe más
   tarde), no como artefacto. Ver decisions.md del proyecto para el detalle
   completo de la investigación.

PENDIENTE, NO RESUELTO EN ESTE ARCHIVO:
- `has_reliable_anchor` usa n≥20 pitches de la recta primaria como umbral
  heurístico SIN derivación empírica — pendiente el split-half check
  (comparar mediana de tiempo de vuelo entre mitades aleatorias de la recta
  primaria, por bin de n, buscar dónde se aplana la discrepancia).
- Pitcher-años sin fastball real en `PRIMARY_FASTBALLS` (la "recta primaria"
  de facto es una breaking ball) quedan como limitación conceptual sin
  resolver — heredada de `_choose_primary_pitch`, no introducida por este fix.
- `plate_y` (posición calculada al cruce de plato) da ~1.7-1.9 ft en vez de
  ~0 — sin explicar. Hipótesis sin confirmar: convención de Trackman
  (borde frontal del plato, ~1.417 ft) vs. y=0, o problema de selección de
  raíz en `_time_to_plate`. No bloqueó este fix, pero resolver antes de
  reportar cualquier cifra de tiempo de vuelo en el documento técnico.
- Leakage de esta feature en pitcher holdout (es relacional, depende del
  arsenal completo del pitcher-año) sigue sin resolverse — el fix de reloj
  no lo toca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reutiliza estas de stuffplus/features.py — NO redefinir si se pega ahí:
# from stuffplus.features import _choose_primary_pitch, _time_to_plate


def build_primary_tunnel_features(data: pd.DataFrame) -> pd.DataFrame:
    """Landmarks release/decisión/plato con el punto de decisión anclado
    a un reloj común por pitcher-año (mediana del tiempo de vuelo de la
    recta primaria), no al tiempo de vuelo propio de cada pitch.

    Cada pitch se evalúa con SU PROPIA ecuación de movimiento
    (release/velocidad/aceleración propios) pero en el MISMO instante de
    reloj que todos los demás pitches del mismo pitcher-año: 150ms antes
    de que la recta primaria (mediana) cruce el plato.
    """
    columns = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]
    required = [c for c in columns if c in data.columns]
    if len(required) != len(columns):
        return pd.DataFrame(index=data.index)

    valid = data[columns].notna().all(axis=1)
    frame = data.loc[valid, columns].copy()
    plate_t_own = pd.Series(_time_to_plate(frame), index=frame.index)

    working = data.loc[valid, ["year", "pitcher_anon_id", "AutoPitchType"]].copy()
    working["plate_t_own"] = plate_t_own

    counts = (
        working.groupby(["year", "pitcher_anon_id", "AutoPitchType"], observed=True)
        .size().rename("n_pitches").reset_index()
    )
    primary_key = _choose_primary_pitch(counts)
    working = working.merge(primary_key, on=["year", "pitcher_anon_id"], how="left")

    primary_rows = working.loc[working["AutoPitchType"].eq(working["primary_pitch_type"])]
    primary_n = primary_rows.groupby(["year", "pitcher_anon_id"], observed=True).size().rename("n_primary")
    primary_clock = (
        primary_rows.groupby(["year", "pitcher_anon_id"], observed=True)["plate_t_own"]
        .median().rename("primary_plate_t").reset_index()
        .merge(primary_n.reset_index(), on=["year", "pitcher_anon_id"])
    )
    working = working.merge(primary_clock, on=["year", "pitcher_anon_id"], how="left")
    working["decision_t_common"] = np.maximum(working["primary_plate_t"] - 0.150, 0.0)

    t = working["decision_t_common"].to_numpy(float)
    output = pd.DataFrame(index=data.index)
    for coordinate in ("x", "y", "z"):
        output.loc[valid, f"decision_{coordinate}"] = (
            frame[f"{coordinate}0"].to_numpy(float)
            + frame[f"v{coordinate}0"].to_numpy(float) * t
            + 0.5 * frame[f"a{coordinate}0"].to_numpy(float) * t**2
        )
        output.loc[valid, f"release_{coordinate}"] = frame[f"{coordinate}0"].to_numpy(float)
        output.loc[valid, f"plate_{coordinate}"] = (
            frame[f"{coordinate}0"].to_numpy(float)
            + frame[f"v{coordinate}0"].to_numpy(float) * plate_t_own.to_numpy(float)
            + 0.5 * frame[f"a{coordinate}0"].to_numpy(float) * plate_t_own.to_numpy(float) ** 2
        )

    output.loc[valid, "time_to_plate_s_own"] = plate_t_own.to_numpy(float)
    output.loc[valid, "decision_t_common_s"] = t
    output.loc[valid, "primary_pitch_type"] = working["primary_pitch_type"].to_numpy()
    output.loc[valid, "n_primary_pitches"] = working["n_primary"].to_numpy()
    # n>=20 es un umbral heurístico, NO derivado empíricamente -- ver pendiente arriba.
    output.loc[valid, "has_reliable_anchor"] = working["n_primary"].fillna(0).ge(20).to_numpy()
    return output


def tunnel_separation_vs_primary(data: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    """Separación (ft) en el plano x-z (lateral + vertical) en el punto de
    decisión, release y plato, vs. la recta primaria del mismo pitcher-año.

    Deliberadamente excluye `y` (profundidad): a un mismo instante de
    reloj, dos pitches de velocidad distinta están a distancias distintas
    del plato por construcción, y esa diferencia de profundidad no es lo
    que el bateador percibe como separación de trayectoria -- mezclarla
    infla artificialmente la separación en el punto de decisión para
    pitches lentos (ver bug documentado arriba).

    Solo `decision_separation_ft` debe entrar como feature de modelo;
    release/plato quedan como diagnóstico y visualización.
    """
    working = pd.concat(
        [data[["year", "pitcher_anon_id", "AutoPitchType"]], landmarks], axis=1
    )
    for point in ("release", "decision", "plate"):
        primary_pos = (
            working.loc[working["AutoPitchType"].eq(working["primary_pitch_type"])]
            .groupby(["year", "pitcher_anon_id"], observed=True)
            [[f"{point}_x", f"{point}_z"]]
            .mean()
            .rename(columns=lambda c: f"primary_{c}")
            .reset_index()
        )
        working = working.merge(primary_pos, on=["year", "pitcher_anon_id"], how="left")
        dx = working[f"{point}_x"] - working[f"primary_{point}_x"]
        dz = working[f"{point}_z"] - working[f"primary_{point}_z"]
        working[f"{point}_separation_ft"] = np.hypot(dx, dz)
    return working
