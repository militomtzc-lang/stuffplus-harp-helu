"""Entrena modelos de IVB/HB y estima residuales no-Magnus por arsenal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.movement import residual_profiles, train_movement_models  # noqa: E402


def _stratified_sample(data: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(data):
        return data
    parts = []
    proportions = data["AutoPitchType"].value_counts(normalize=True)
    for pitch_type, proportion in proportions.items():
        group = data.loc[data["AutoPitchType"].eq(pitch_type)]
        n = min(len(group), max(1, round(sample_size * proportion)))
        parts.append(group.sample(n=n, random_state=42))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="physics_features.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--shrinkage-k", type=float, default=100.0)
    args = parser.parse_args()

    raw = pd.read_parquet(args.input)
    raw["altitude_category"] = raw["altitude_category"].where(raw["altitude_category"].notna())
    n_before = len(raw)
    raw = raw.dropna(subset=["altitude_category"])
    n_dropped = n_before - len(raw)
    print(
        f"Descartadas {n_dropped:,} filas sin altitude_category "
        f"({n_dropped / n_before:.3%}) antes de entrenar."
    )

    data = _stratified_sample(raw, args.sample_size)
    models, oof, metrics = train_movement_models(data, n_splits=args.n_splits)
    profiles = residual_profiles(oof, shrinkage_k=args.shrinkage_k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(output_dir / "movement_oof_predictions.parquet", index=False)
    profiles.to_csv(output_dir / "non_magnus_residual_profiles.csv", index=False)
    joblib.dump(models, output_dir / "movement_models.joblib")
    (output_dir / "movement_metrics.json").write_text(
        json.dumps(
            {
                "metrics_pitcher_oof": metrics,
                "definition": "Residual = movimiento observado - predicción spin/shape OOF.",
                "warning": "El residual es compatible con SSW, pero no lo mide directamente.",
                "shrinkage_k": args.shrinkage_k,
                "rows_dropped_missing_altitude_category": int(n_dropped),
                "rows_dropped_pct": round(n_dropped / n_before, 5),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
