"""Lectura y validación mínima de datos pitch-by-pitch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ID_COLUMNS = ["PitchUID", "game_anon_id", "pitcher_anon_id"]
REQUIRED_COLUMNS = {
    "PitchUID",
    "year",
    "pitcher_anon_id",
    "AutoPitchType",
    "RelSpeed",
    "SpinRate",
    "InducedVertBreak",
    "HorzBreak",
    "RelHeight",
    "RelSide",
    "Extension",
    "is_whiff",
    "is_called_strike",
    "is_batted",
    "ExitSpeed",
}


def read_pitch_data(path: str | Path) -> pd.DataFrame:
    """Lee parquet o pickle sin modificar el archivo original."""
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
    """Corrige tipos evidentes sin imputar ni crear targets."""
    data = frame.copy()
    numeric = [
        "RelSpeed", "EffectiveVelo", "ZoneSpeed", "SpinRate", "SpinAxis",
        "RelHeight", "RelSide", "Extension", "VertBreak", "InducedVertBreak",
        "HorzBreak", "VertRelAngle", "HorzRelAngle", "VertApprAngle",
        "HorzApprAngle", "SpeedDrop", "ZoneTime", "x0", "y0", "z0", "vx0",
        "vy0", "vz0", "ax0", "ay0", "az0", "ExitSpeed",
    ]
    for column in numeric:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["year", "Balls", "Strikes", "Outs", "Inning"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Fix Bug 2 (sesion 17 sept): eje x del feed de ISAC (x0, vx0, ax0) definido
    # con orientacion opuesta a RelSide/HorzBreak/PlateLocSide. Verificado contra
    # PlateLocSide real: std_error 22.29in -> 0.63in tras negar.
    for column in ["x0", "vx0", "ax0"]:
        if column in data:
            data[column] = -data[column]

    return data
