"""Features de Stuff físico y diagnóstico de arsenal/túnel."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


PHYSICAL_NUMERIC = [
    "RelSpeed", "EffectiveVelo", "ZoneSpeed", "SpinRate", "SpinAxis",
    "RelHeight", "RelSide", "Extension", "VertBreak", "InducedVertBreak",
    "HorzBreak", "VertRelAngle", "HorzRelAngle", "VertApprAngle",
    "HorzApprAngle", "SpeedDrop", "ZoneTime",
]
PHYSICAL_CATEGORICAL = ["AutoPitchType", "PitcherThrows"]
PRIMARY_FASTBALLS = ("Four-Seam", "Sinker", "Cutter", "OneSeamFastBall", "TwoSeamFastBall")


def make_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Targets por lanzamiento; no se reutilizan como features."""
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


def _choose_primary_pitch(profile: pd.DataFrame) -> pd.DataFrame:
    """Escoge Four-Seam cuando existe; si no, el fastball más usado."""
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
    """Agrega diferenciales contra la recta primaria y frecuencia de uso.

    Cada perfil usa solamente pitcher, temporada y tipo de pitch. Los IDs no se
    entregan al modelo y solo sirven para construir agregados relacionales.
    """
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


def stuff_feature_columns(include_altitude: bool) -> tuple[list[str], list[str]]:
    """Regresa columnas permitidas para el modelo de Stuff físico."""
    numerical = PHYSICAL_NUMERIC + [
        "pitch_usage_pct", "relspeed_diff_vs_primary", "inducedvertbreak_diff_vs_primary",
        "horzbreak_diff_vs_primary", "relheight_diff_vs_primary",
        "relside_diff_vs_primary", "extension_diff_vs_primary", "release_gap_vs_primary",
    ]
    categorical = PHYSICAL_CATEGORICAL.copy()
    if include_altitude:
        categorical.append("altitude_category")
    return numerical, categorical


def _time_to_plate(data: pd.DataFrame) -> np.ndarray:
    """Resuelve y(t)=0 con coordenadas Trackman; retorna NaN si no hay raíz física.

    Fix Bug 1 (sesion 16-17 sept): el fallback lineal (-y0/vy, ignora ay0) ya NO
    compite por el minimo contra las raices cuadraticas correctas. Solo se usa
    si la cuadratica es degenerada y no produjo ninguna raiz positiva valida.
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


def trajectory_landmarks(data: pd.DataFrame) -> pd.DataFrame:
    """Posiciones aproximadas en release, decisión y plato.

    El punto de decisión se toma 150 ms antes del plato. La posición de plato
    conserva la ubicación real, por lo que es descriptiva y no entra a Stuff+.
    """
    columns = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]
    required = [c for c in columns if c in data.columns]
    if len(required) != len(columns):
        return pd.DataFrame(index=data.index)
    valid = data[columns].notna().all(axis=1)
    frame = data.loc[valid, columns].copy()
    plate_t = _time_to_plate(frame)
    decision_t = np.maximum(plate_t - 0.150, 0.0)
    output = pd.DataFrame(index=data.index)
    for prefix, times in {"release": np.zeros(len(frame)), "decision": decision_t, "plate": plate_t}.items():
        for coordinate in ("x", "y", "z"):
            output.loc[valid, f"{prefix}_{coordinate}"] = (
                frame[f"{coordinate}0"].to_numpy(float)
                + frame[f"v{coordinate}0"].to_numpy(float) * times
                + 0.5 * frame[f"a{coordinate}0"].to_numpy(float) * times**2
            )
    output.loc[valid, "time_to_plate_s"] = plate_t
    return output


def build_arsenal_overlap(data: pd.DataFrame) -> pd.DataFrame:
    """Compara trayectorias promedio de cada par de tipos por pitcher-temporada."""
    landmarks = trajectory_landmarks(data)
    working = pd.concat([data[["year", "pitcher_anon_id", "AutoPitchType"]], landmarks], axis=1)
    coord_cols = [f"{point}_{axis}" for point in ("release", "decision", "plate") for axis in "xyz"]
    profiles = (
        working.groupby(["year", "pitcher_anon_id", "AutoPitchType"], dropna=False, observed=True)[coord_cols]
        .mean()
        .reset_index()
    )
    profile_counts = (
        working.groupby(["year", "pitcher_anon_id", "AutoPitchType"], dropna=False, observed=True)
        .size()
        .rename("n_pitches")
        .reset_index()
    )
    profiles = profiles.merge(profile_counts, on=["year", "pitcher_anon_id", "AutoPitchType"], how="left")
    records: list[dict[str, object]] = []
    for (year, pitcher), group in profiles.groupby(["year", "pitcher_anon_id"], dropna=False):
        rows = group.dropna(subset=["AutoPitchType"])
        for left, right in combinations(rows.to_dict("records"), 2):
            record: dict[str, object] = {
                "year": year,
                "pitcher_anon_id": pitcher,
                "pitch_type_a": left["AutoPitchType"],
                "pitch_type_b": right["AutoPitchType"],
                "n_pitches_a": left["n_pitches"],
                "n_pitches_b": right["n_pitches"],
            }
            for point in ("release", "decision", "plate"):
                delta = np.array([left[f"{point}_{axis}"] - right[f"{point}_{axis}"] for axis in "xyz"])
                record[f"{point}_separation_ft"] = float(np.linalg.norm(delta))
            records.append(record)
    return pd.DataFrame.from_records(records)
