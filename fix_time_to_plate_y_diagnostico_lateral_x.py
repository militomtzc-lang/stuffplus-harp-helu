"""Bug 1 (fix integrado) + diagnóstico Bug 2 (error lateral en x) — sesión 17 sept 2026.

Pensado para pegar en notebook, celda por celda (no CLI). Dos partes:

PARTE A — reemplazo de `_time_to_plate` en `stuffplus/features.py`.
  El fallback lineal (-y0/vy, ignora ay0) ya NO compite por el mínimo contra
  las raíces cuadráticas correctas. Solo se usa si la cuadrática es
  degenerada (ay0≈0) Y no produce ninguna raíz positiva válida. Antes del
  fix, el lineal ganaba el min() el 100% de las veces porque, al ignorar la
  desaceleración, SIEMPRE subestima el tiempo real de vuelo.

PARTE B — diagnóstico del Bug 2 (std(diff_x) ≈ 21-22 in vs. PlateLocSide),
  en tres hipótesis ordenadas de más barata a más cara:

  H_C (chequeo de entrada, antes que cualquier física de vuelo):
      ¿x0 ya es consistente con RelSide sin integrar tiempo? Si no, el
      problema es de definición de columna, no de `_time_to_plate`.
      Pista ya documentada en stuff_plus_data_dictionary.csv: x0 se mide
      "from the center of the rubber", y0/z0 "from the tip of home plate"
      -- origen distinto al del resto de la trayectoria. Consistente con
      que z (mismo origen que y) haya salido limpio y x no.

  H_A (offset de origen en x, motivada por lo anterior):
      un desfase constante de origen debería verse como sesgo (mean≠0),
      no necesariamente como varianza grande -- probar igual, pero no
      esperar que sea la explicación completa del std.

  H_B (signo de ax0 invertido respecto a HorzBreak/vx0):
      mecanismo que sí predice la MAGNITUD observada: si el término de
      aceleración lateral resta en vez de sumar quiebre, el error de
      posición final es ~2x el HorzBreak real -- para quiebres típicos de
      10-12in eso da std de posición de ~20-24in. Encaja mejor con lo
      reportado que un offset fijo.

No corrido contra datos reales en esta sesión (sin acceso al .pkl/.parquet
de ISAC en este entorno) -- correr con el dataset real y traer los outputs
de las tres funciones de diagnóstico a la próxima sesión.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PARTE A — reemplazo directo de `_time_to_plate` en stuffplus/features.py
# ---------------------------------------------------------------------------
def _time_to_plate(data: pd.DataFrame) -> np.ndarray:
    """Resuelve y(t)=0 con coordenadas Trackman; retorna NaN si no hay raíz física.

    Fix Bug 1: el fallback lineal ya no compite por el mínimo contra las
    raíces cuadráticas -- solo se usa si la cuadrática es degenerada
    (|ay0/2| < 1e-6) y no produjo ninguna raíz positiva finita.
    """
    y0 = data["y0"].to_numpy(float)
    vy = data["vy0"].to_numpy(float)
    half_ay = 0.5 * data["ay0"].to_numpy(float)
    discriminant = vy**2 - 4 * half_ay * y0
    with np.errstate(invalid="ignore", divide="ignore"):
        root1 = (-vy + np.sqrt(discriminant)) / (2 * half_ay)
        root2 = (-vy - np.sqrt(discriminant)) / (2 * half_ay)
    quad_roots = np.column_stack([root1, root2])
    quad_roots[~np.isfinite(quad_roots) | (quad_roots <= 0)] = np.inf
    quad_earliest = quad_roots.min(axis=1)

    degenerate = np.abs(half_ay) < 1e-6
    linear = np.where(vy != 0, -y0 / vy, np.nan)
    linear_valid = np.isfinite(linear) & (linear > 0)
    use_linear = degenerate & ~np.isfinite(quad_earliest) & linear_valid

    earliest = quad_earliest.copy()
    earliest[use_linear] = linear[use_linear]
    earliest[~np.isfinite(earliest)] = np.nan
    return earliest


# Checklist de verificación post-integración (correr antes de confiar en nada más):
#   1. chosen_branch debe dejar de ser "linear" el 100% de las veces
#      (confirmar sobre la misma muestra de 50k filas usada la sesión pasada).
#   2. Re-correr `tunnel_features_fix.py` completo (build_primary_tunnel_features +
#      tunnel_separation_vs_primary) con este _time_to_plate y comparar
#      inversion_rate contra el resultado ya reportado (4-23%, Curveball 38.8%).
#      Si el patrón NO sobrevive, no se puede asumir "sesgo parejo entre tipos"
#      como se especuló -- hay que reportar el número nuevo, no el viejo.
#   3. plate_y (posición reconstruida en el cruce) debería acercarse a 0
#      si el bug de plate_y≈1.7-1.9ft era enteramente este bug, como se
#      documentó. Si sigue habiendo un residuo, no cerrar ese pendiente todavía.


# ---------------------------------------------------------------------------
# PARTE B — diagnóstico del Bug 2 (error lateral en x)
# ---------------------------------------------------------------------------
def diagnose_input_consistency(data: pd.DataFrame) -> pd.DataFrame:
    """H_C -- ¿x0/z0 ya son consistentes con RelSide/RelHeight SIN integrar tiempo?

    Si x0 no correlaciona ~1:1 con RelSide (a diferencia de z0 con RelHeight),
    el problema vive en la columna de entrada, no en `_time_to_plate` -- ningún
    fix de tiempo de vuelo lo va a arreglar.
    """
    checks = {}
    for coord, release_col in [("x0", "RelSide"), ("z0", "RelHeight")]:
        a = pd.to_numeric(data[coord], errors="coerce")
        b = pd.to_numeric(data[release_col], errors="coerce")
        valid = a.notna() & b.notna()
        diff = a[valid] - b[valid]
        checks[f"{coord}_vs_{release_col}"] = {
            "corr": float(a[valid].corr(b[valid])),
            "diff_mean_ft": float(diff.mean()),
            "diff_std_ft": float(diff.std()),
            "n": int(valid.sum()),
        }
    return pd.DataFrame(checks).T


def diagnose_ax0_sign_vs_horzbreak(data: pd.DataFrame) -> dict:
    """H_B directo -- ¿ax0 tiene el signo correcto respecto a HorzBreak (ya confiable)?

    Una correlación negativa fuerte es evidencia directa de signo invertido
    en el término de aceleración lateral.
    """
    ax0 = pd.to_numeric(data["ax0"], errors="coerce")
    hb = pd.to_numeric(data["HorzBreak"], errors="coerce")
    valid = ax0.notna() & hb.notna()
    return {
        "corr_ax0_horzbreak": float(ax0[valid].corr(hb[valid])),
        "n": int(valid.sum()),
    }


def diagnose_lateral_sign_variants(data: pd.DataFrame, plate_t: np.ndarray) -> pd.DataFrame:
    """H_A / H_B comparados directamente contra PlateLocSide (ground truth de ISAC).

    plate_t: array de tiempo de vuelo por fila, ya calculado con el
    `_time_to_plate` corregido de la Parte A (pasar el output de esa función,
    alineado por índice con `data`).

    La variante con mayor corr y menor std(diff) señala el mecanismo real.
    No asuman de antemano cuál gana -- ese es el punto del diagnóstico.
    """
    x0 = data["x0"].to_numpy(float)
    vx0 = data["vx0"].to_numpy(float)
    ax0 = data["ax0"].to_numpy(float)
    truth = pd.to_numeric(data["PlateLocSide"], errors="coerce").to_numpy(float)
    t = np.asarray(plate_t, dtype=float)

    variants = {
        "baseline (sin cambios)": x0 + vx0 * t + 0.5 * ax0 * t**2,
        "flip_ax0_solo (H_B)": x0 + vx0 * t - 0.5 * ax0 * t**2,
        "flip_x0_solo (H_A, offset de signo)": -x0 + vx0 * t + 0.5 * ax0 * t**2,
        "flip_vx0_y_ax0": x0 - vx0 * t - 0.5 * ax0 * t**2,
        "flip_todo": -(x0 + vx0 * t + 0.5 * ax0 * t**2),
    }
    rows = []
    for name, computed in variants.items():
        diff = computed - truth
        valid = np.isfinite(diff)
        corr = np.corrcoef(computed[valid], truth[valid])[0, 1] if valid.sum() > 1 else np.nan
        rows.append(
            {
                "variante": name,
                "corr_vs_PlateLocSide": float(corr),
                "diff_mean_in": float(np.nanmean(diff[valid]) * 12),
                "diff_std_in": float(np.nanstd(diff[valid]) * 12),
                "n": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("diff_std_in")


# ---------------------------------------------------------------------------
# Orden de ejecución sugerido (celdas de notebook)
# ---------------------------------------------------------------------------
# data = normalize_types(read_pitch_data("<ruta al .pkl/.parquet de ISAC>"))
#
# # 1. H_C primero -- el chequeo más barato y el que debieron correr antes
# #    de descartar mano de pitcher/bateador.
# print(diagnose_input_consistency(data))
#
# # 2. H_B directo, sin siquiera tocar tiempo de vuelo.
# print(diagnose_ax0_sign_vs_horzbreak(data))
#
# # 3. Con el fix de la Parte A ya integrado en features.py:
# from stuffplus.features import trajectory_landmarks  # después de pegar el fix
# plate_t = _time_to_plate(data.loc[data[["x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0"]].notna().all(axis=1)])
# # (alinear índices con cuidado -- ver cómo lo hace trajectory_landmarks)
# print(diagnose_lateral_sign_variants(data, plate_t))
