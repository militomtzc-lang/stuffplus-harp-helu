"""Simula movimiento para una referencia de altitud y Harp Helú."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from movement import predict_movement
from physics import simulate_density_scaled_movement


def _filter_data(data: pd.DataFrame, pitcher: str | None, year: int | None) -> pd.DataFrame:
    if pitcher is not None:
        data = data.loc[data["pitcher_anon_id"].eq(pitcher)]
    if year is not None:
        data = data.loc[data["year"].eq(year)]
    if data.empty:
        raise ValueError("El filtro no encontró lanzamientos.")
    return data.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="physics_features.parquet")
    parser.add_argument("--models", required=True, help="movement_models.joblib")
    parser.add_argument("--residual-profiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitcher", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--reference-altitude-m", type=float, default=0.0)
    parser.add_argument("--target-altitude-m", type=float, default=2240.0)
    parser.add_argument("--reference-category", default="No Altitude")
    parser.add_argument("--target-category", default="Extreme Altitude")
    args = parser.parse_args()

    data = _filter_data(pd.read_parquet(args.input), args.pitcher, args.year)
    models = joblib.load(args.models)
    profiles = pd.read_csv(args.residual_profiles)
    keys = ["year", "pitcher_anon_id", "AutoPitchType"]
    data = data.merge(
        profiles[keys + ["n_pitches", "ivb_non_magnus_residual_shrunk", "hb_non_magnus_residual_shrunk"]],
        on=keys,
        how="left",
    )
    data[["ivb_non_magnus_residual_shrunk", "hb_non_magnus_residual_shrunk"]] = data[
        ["ivb_non_magnus_residual_shrunk", "hb_non_magnus_residual_shrunk"]
    ].fillna(0.0)

    reference = data.copy()
    reference["altitude_category"] = args.reference_category
    target_category = data.copy()
    target_category["altitude_category"] = args.target_category
    reference_pred = predict_movement(models, reference)
    category_pred = predict_movement(models, target_category)

    conservative = simulate_density_scaled_movement(
        reference_pred["ivb_spin_shape_pred"],
        reference_pred["hb_spin_shape_pred"],
        data["ivb_non_magnus_residual_shrunk"],
        data["hb_non_magnus_residual_shrunk"],
        target_altitude_m=args.target_altitude_m,
        reference_altitude_m=args.reference_altitude_m,
        ssw_density_scale=0.0,
    ).rename(columns={"simulated_ivb": "harp_ivb_conservative", "simulated_hb": "harp_hb_conservative"})
    proportional = simulate_density_scaled_movement(
        reference_pred["ivb_spin_shape_pred"],
        reference_pred["hb_spin_shape_pred"],
        data["ivb_non_magnus_residual_shrunk"],
        data["hb_non_magnus_residual_shrunk"],
        target_altitude_m=args.target_altitude_m,
        reference_altitude_m=args.reference_altitude_m,
        ssw_density_scale=1.0,
    ).rename(columns={"simulated_ivb": "harp_ivb_proportional", "simulated_hb": "harp_hb_proportional"})

    output = data[keys + ["PitchUID", "altitude_category", "n_pitches"]].copy()
    output["sea_level_ivb_spin_shape"] = reference_pred["ivb_spin_shape_pred"].to_numpy()
    output["sea_level_hb_spin_shape"] = reference_pred["hb_spin_shape_pred"].to_numpy()
    output["harp_ivb_category_model"] = category_pred["ivb_spin_shape_pred"].to_numpy()
    output["harp_hb_category_model"] = category_pred["hb_spin_shape_pred"].to_numpy()
    # Ambos escenarios usan la misma razón de densidad. Se conserva una sola
    # columna para que Parquet no reciba nombres duplicados.
    proportional = proportional.drop(columns=["air_density_ratio"])
    output = pd.concat([output.reset_index(drop=True), conservative.reset_index(drop=True), proportional.reset_index(drop=True)], axis=1)
    output["delta_ivb_conservative"] = output["harp_ivb_conservative"] - output["sea_level_ivb_spin_shape"]
    output["delta_hb_conservative"] = output["harp_hb_conservative"] - output["sea_level_hb_spin_shape"]
    output["delta_ivb_proportional"] = output["harp_ivb_proportional"] - output["sea_level_ivb_spin_shape"]
    output["delta_hb_proportional"] = output["harp_hb_proportional"] - output["sea_level_hb_spin_shape"]
    output.to_parquet(args.output, index=False)
    print(f"Simuladas {len(output):,} filas. Archivo: {args.output}")


if __name__ == "__main__":
    main()
