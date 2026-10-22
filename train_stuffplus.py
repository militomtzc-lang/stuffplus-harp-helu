"""Entrena una línea base física de Stuff+ y diagnósticos de arsenal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import normalize_types, read_pitch_data
from features import (
    add_arsenal_features,
    build_arsenal_overlap,
    make_targets,
    stuff_feature_columns,
)
from modeling import altitude_counterfactual, fit_validate_and_score, score_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Ruta al parquet o pickle entregado por el hackathon.")
    parser.add_argument("--output-dir", required=True, help="Carpeta local segura para resultados y modelos.")
    parser.add_argument("--sample-size", type=int, default=None, help="Muestra aleatoria para prototipar rápido.")
    parser.add_argument(
        "--include-altitude",
        action="store_true",
        help="Incluye altitude_category. Permite un contraste asociativo, no causal.",
    )
    return parser.parse_args()


def _sample(data: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(data):
        return data
    # El muestreo estratificado por tipo evita perder pitches poco frecuentes.
    proportions = data["AutoPitchType"].value_counts(normalize=True)
    pieces = []
    for pitch_type, proportion in proportions.items():
        group = data.loc[data["AutoPitchType"].eq(pitch_type)]
        size = min(len(group), max(1, round(sample_size * proportion)))
        pieces.append(group.sample(n=size, random_state=42))
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = normalize_types(read_pitch_data(args.input))
    data = _sample(data, args.sample_size)
    data = make_targets(data)
    data = add_arsenal_features(data)
    numerical, categorical = stuff_feature_columns(args.include_altitude)

    model, holdout, metrics = fit_validate_and_score(data, numerical, categorical)
    scores = score_all(model, data)
    scores["extreme_minus_no_altitude_stuff_plus"] = altitude_counterfactual(model, data).to_numpy()

    pitcher_summary = (
        scores.groupby(["year", "pitcher_anon_id"], as_index=False, observed=True)
        .agg(pitches=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean"))
        .sort_values("stuff_plus", ascending=False)
    )
    arsenal_summary = (
        scores.groupby(["year", "pitcher_anon_id", "AutoPitchType"], as_index=False, observed=True)
        .agg(
            pitches=("PitchUID", "size"),
            stuff_plus=("stuff_plus", "mean"),
            altitude_delta=("extreme_minus_no_altitude_stuff_plus", "mean"),
        )
        .sort_values("stuff_plus", ascending=False)
    )
    overlap = build_arsenal_overlap(data)

    holdout.to_csv(output_dir / "holdout_pitch_scores.csv", index=False)
    scores.to_parquet(output_dir / "pitch_scores.parquet", index=False)
    pitcher_summary.to_csv(output_dir / "pitcher_summary.csv", index=False)
    arsenal_summary.to_csv(output_dir / "arsenal_summary.csv", index=False)
    overlap.to_csv(output_dir / "arsenal_overlap.csv", index=False)
    joblib.dump(model, output_dir / "stuffplus_model.joblib")
    metadata = {
        "rows_scored": int(len(scores)),
        "sample_size": args.sample_size,
        "include_altitude": args.include_altitude,
        "metrics_pitcher_holdout": metrics,
        "score_definition": "100 + 10 * z(0.45*P(whiff) + 0.15*P(called_strike) + 0.15*P(groundball) - 0.25*P(hard_contact))",
        "limitation": "Extreme Altitude minus No Altitude is an associational model contrast, not a Harp Helu-specific causal effect.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({"rows_scored": len(scores), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
