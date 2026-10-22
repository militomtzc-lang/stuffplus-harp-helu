"""Genera un mapa release-decisión-plato para un pitcher y temporada."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.data import normalize_types, read_pitch_data  # noqa: E402
from stuffplus.features import trajectory_landmarks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pitcher", required=True, help="ID anonimizado del pitcher.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", required=True, help="Archivo PNG de salida.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = normalize_types(read_pitch_data(args.input))
    subset = data.loc[
        data["pitcher_anon_id"].eq(args.pitcher) & data["year"].eq(args.year)
    ].copy()
    if subset.empty:
        raise ValueError("No hay lanzamientos para ese pitcher y temporada.")

    landmarks = trajectory_landmarks(subset)
    profile = pd.concat([subset[["AutoPitchType"]], landmarks], axis=1).groupby(
        "AutoPitchType", observed=True
    ).mean(numeric_only=True)
    if profile.empty:
        raise ValueError("Faltan coordenadas cinemáticas para este pitcher-temporada.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False, sharey=False)
    labels = [("release", "Release"), ("decision", "Decisión de swing (150 ms)"), ("plate", "Plato")]
    colors = plt.get_cmap("tab10")
    for axis, (prefix, title) in zip(axes, labels):
        for index, (pitch_type, row) in enumerate(profile.iterrows()):
            axis.scatter(row[f"{prefix}_x"], row[f"{prefix}_z"], s=75, color=colors(index % 10))
            axis.annotate(str(pitch_type), (row[f"{prefix}_x"], row[f"{prefix}_z"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.axhline(0, color="0.85", lw=1)
        axis.axvline(0, color="0.85", lw=1)
        axis.set_title(title)
        axis.set_xlabel("Horizontal (ft)")
        axis.set_ylabel("Altura (ft)")
        axis.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"Arsenal: {args.pitcher}, {args.year}")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
