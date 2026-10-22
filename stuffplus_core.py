"""stuffplus_core.py — módulo consolidado, con los fixes de Bug 1 y Bug 2 ya
integrados de fábrica. Reemplaza el uso disperso de data.py/features.py para
todo lo relacionado con tunneling y features de arsenal.

Bug 1 (tiempo de vuelo): el fallback lineal en _time_to_plate ignoraba la
desaceleración y por eso siempre ganaba el min() contra las raíces cuadráticas
correctas — subestimaba el tiempo de vuelo real en ~30ms en promedio.

Bug 2 (eje x reflejado): x0/vx0/ax0 del feed de ISAC vienen con orientación
opuesta a RelSide/HorzBreak/PlateLocSide. Confirmado contra PlateLocSide real:
std_error 22.29in -> 0.63in tras negar.

Este archivo NO modifica tus data.py/features.py originales — es una versión
paralela, limpia, para no arrastrar el estado confuso del kernel. Cuando el
equipo confíe en esto, se puede reemplazar el contenido de data.py/features.py
del repo por lo de aquí.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# I/O y normalización (data.py + fix Bug 2)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "PitchUID", "year", "pitcher_anon_id", "AutoPitchType", "RelSpeed",
    "SpinRate", "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide",
    "Extension", "is_whiff", "is_called_strike", "is_batted", "ExitSpeed",
}

NUMERIC_COLUMNS = [
    "RelSpeed", "EffectiveVelo", "ZoneSpeed", "SpinRate", "SpinAxis",
    "RelHeight", "RelSide", "Extension", "VertBreak", "InducedVertBreak",
    "HorzBreak", "VertRelAngle", "HorzRelAngle", "VertApprAngle",
    "HorzApprAngle", "SpeedDrop", "ZoneTime", "x0", "y0", "z0", "vx0",
    "vy0", "vz0", "ax0", "ay0", "az0", "ExitSpeed",
]


def read_pitch_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        raise ValueError("El input debe ser .parquet, .pq, .pkl o .pickle")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {sorted(missing)}")
    return frame.copy()


def normalize_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos evidentes. Incluye el fix del Bug 2 (eje x reflejado)."""
    data = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["year", "Balls", "Strikes", "Outs", "Inning"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # --- FIX BUG 2 ---
    for column in ["x0", "vx0", "ax0"]:
        if column in data:
            data[column] = -data[column]

    return data


# ---------------------------------------------------------------------------
# Targets (features.py, sin cambios respecto al original)
# ---------------------------------------------------------------------------
def make_targets(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["target_whiff"] = pd.to_numeric(result["is_whiff"], errors="coerce").fillna(0).astype(int)
    result["target_called_strike"] = (
        pd.to_numeric(result["is_called_strike"], errors="coerce").fillna(0).astype(int)
    )
    hit_type = result.get("hit_type", pd.Series(index=result.index, dtype="object")).astype(str).str.lower()
    in_play = pd.to_numeric(result["is_ball_in_play"], errors="coerce").fillna(0).eq(1)
    result["target_groundball"] = (in_play & hit_type.str.contains("ground", na=False)).astype(int)
    batted = pd.to_numeric(result["is_batted"], errors="coerce").fillna(0).eq(1)
    exit_speed = pd.to_numeric(result["ExitSpeed"], errors="coerce")
    result["target_hard_contact"] = (in_play & batted & exit_speed.ge(95)).astype(int)
    return result


# ---------------------------------------------------------------------------
# Arsenal / recta primaria (features.py, sin cambios)
# ---------------------------------------------------------------------------
PRIMARY_FASTBALLS = ("Four-Seam", "Sinker", "Cutter", "OneSeamFastBall", "TwoSeamFastBall")


def _choose_primary_pitch(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.copy()
    profile["is_fastball"] = profile["AutoPitchType"].isin(PRIMARY_FASTBALLS)
    profile["fastball_rank"] = profile["is_fastball"].astype(int)
    ordered = profile.sort_values(
        ["year", "pitcher_anon_id", "fastball_rank", "n_pitches"],
        ascending=[True, True, False, False],
    )
    primary = ordered.groupby(["year", "pitcher_anon_id"], as_index=False).first()
    return primary[["year", "pitcher_anon_id", "AutoPitchType"]].rename(
        columns={"AutoPitchType": "primary_pitch_type"}
    )


def add_arsenal_features(data: pd.DataFrame) -> pd.DataFrame:
    grouping = ["year", "pitcher_anon_id", "AutoPitchType"]
    measures = ["RelSpeed", "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide", "Extension"]
    profile = (
        data.groupby(grouping, dropna=False, observed=True)[measures]
        .mean()
        .rename(columns={name: f"profile_{name}" for name in measures})
        .reset_index()
    )
    counts = data.groupby(grouping, dropna=False, observed=True).size().rename("n_pitches").reset_index()
    profile = profile.merge(counts, on=grouping, how="left")
    total = profile.groupby(["year", "pitcher_anon_id"])["n_pitches"].transform("sum")
    profile["pitch_usage_pct"] = profile["n_pitches"] / total

    primary_key = _choose_primary_pitch(profile)
    profile = profile.merge(primary_key, on=["year", "pitcher_anon_id"], how="left")
    primary = profile.loc[
        profile["AutoPitchType"].eq(profile["primary_pitch_type"]),
        ["year", "pitcher_anon_id", *[f"profile_{m}" for m in measures]],
    ].copy()
    primary = primary.rename(columns={f"profile_{m}": f"primary_{m}" for m in measures})
    profile = profile.merge(primary, on=["year", "pitcher_anon_id"], how="left")
    for measure in measures:
        profile[f"{measure.lower()}_diff_vs_primary"] = (
            profile[f"profile_{measure}"] - profile[f"primary_{measure}"]
        )
    release_dx = profile["profile_RelSide"] - profile["primary_RelSide"]
    release_dz = profile["profile_RelHeight"] - profile["primary_RelHeight"]
    profile["release_gap_vs_primary"] = np.hypot(release_dx, release_dz)

    added = [
        "primary_pitch_type", "pitch_usage_pct", "relspeed_diff_vs_primary",
        "inducedvertbreak_diff_vs_primary", "horzbreak_diff_vs_primary",
        "relheight_diff_vs_primary", "relside_diff_vs_primary",
        "extension_diff_vs_primary", "release_gap_vs_primary",
    ]
    return data.merge(profile[grouping + added], on=grouping, how="left")


# ---------------------------------------------------------------------------
# Tiempo de vuelo — FIX BUG 1
# ---------------------------------------------------------------------------
def _time_to_plate(data: pd.DataFrame) -> np.ndarray:
    """Resuelve y(t)=0. El fallback lineal ya NO compite por el mínimo contra
    las raíces cuadráticas correctas — solo se usa si la cuadrática es
    degenerada y no produjo ninguna raíz positiva válida."""
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


# ---------------------------------------------------------------------------
# Tunneling — reloj común anclado a la recta primaria, separación x-z
# (tunnel_features_fix.py, usando el _time_to_plate ya corregido de arriba)
# ---------------------------------------------------------------------------
def build_primary_tunnel_features(data: pd.DataFrame) -> pd.DataFrame:
    """Landmarks release/decisión/plato con decisión anclada a un reloj común
    por pitcher-año: mediana del tiempo de vuelo de la recta primaria."""
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
    output.loc[valid, "has_reliable_anchor"] = working["n_primary"].fillna(0).ge(20).to_numpy()
    return output


def tunnel_separation_vs_primary(data: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    """Separación (ft) en el plano x-z vs. la recta primaria, en release,
    decisión y plato. Solo decision_separation_ft debe usarse como feature."""
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


# ---------------------------------------------------------------------------
# Verificación EMPÍRICA de los fixes — contra datos reales, no contra texto
# ---------------------------------------------------------------------------
def verify_fixes(data: pd.DataFrame) -> dict:
    """Corre esto sobre `data` YA pasado por normalize_types de este módulo.
    Reconstruye la posición en el plato y la compara contra PlateLocSide/
    PlateLocHeight reales de ISAC. Si los fixes están activos, corr debe ser
    ~0.999 y std_error_in debe ser ~0.6-1.5 pulgadas, NO ~20+ pulgadas."""
    columns = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]
    valid = data[columns].notna().all(axis=1) & data["PlateLocSide"].notna()
    frame = data.loc[valid, columns].copy()
    t = _time_to_plate(frame)

    computed_x = frame["x0"].to_numpy(float) + frame["vx0"].to_numpy(float) * t + 0.5 * frame["ax0"].to_numpy(float) * t**2
    computed_z = frame["z0"].to_numpy(float) + frame["vz0"].to_numpy(float) * t + 0.5 * frame["az0"].to_numpy(float) * t**2
    truth_x = data.loc[valid, "PlateLocSide"].to_numpy(float)

    diff_x_in = (computed_x - truth_x) * 12
    result = {
        "n": int(valid.sum()),
        "corr_x_vs_PlateLocSide": float(np.corrcoef(computed_x, truth_x)[0, 1]),
        "mean_error_in": float(np.nanmean(diff_x_in)),
        "std_error_in": float(np.nanstd(diff_x_in)),
        "mean_time_to_plate_s": float(np.nanmean(t)),
    }
    if "PlateLocHeight" in data.columns:
        truth_z = data.loc[valid, "PlateLocHeight"].to_numpy(float)
        diff_z_in = (computed_z - truth_z) * 12
        result["std_error_z_in"] = float(np.nanstd(diff_z_in))
    return result


# ---------------------------------------------------------------------------
# Descomposición de varianza — empirical Bayes (para k y para band_width)
# ---------------------------------------------------------------------------
def variance_components(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Método de momentos, one-way ANOVA no balanceado.
    var_within: varianza dentro de cada grupo (ruido pitch-a-pitch).
    var_between: varianza entre grupos, corregida por ruido de muestreo.
    Devuelve var_between truncada en 0 (no se permite varianza negativa)."""
    grand_mean = df[value_col].mean()
    groups = df.groupby(group_col, observed=True)[value_col]
    n_g = groups.size()
    means_g = groups.mean()
    N = n_g.sum()
    G = len(n_g)

    ss_within = groups.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
    df_within = N - G
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    ss_between = (n_g * (means_g - grand_mean) ** 2).sum()
    df_between = G - 1
    ms_between = ss_between / df_between if df_between > 0 else np.nan

    n_bar = (N - (n_g ** 2).sum() / N) / df_between if df_between > 0 else np.nan
    tau2 = max((ms_between - ms_within) / n_bar, 0.0) if np.isfinite(n_bar) and n_bar > 0 else np.nan

    icc = tau2 / (tau2 + ms_within) if np.isfinite(tau2) and (tau2 + ms_within) > 0 else np.nan
    return {"var_within": ms_within, "var_between": tau2, "icc": icc, "n_groups": G, "n_obs": int(N)}
