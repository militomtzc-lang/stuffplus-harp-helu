"""Prepara variables físicas para el modelo de Magnus/SSW y altitud."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.data import normalize_types, read_pitch_data  # noqa: E402
from stuffplus.physics import add_physics_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-altitude-m", type=float, default=0.0)
    parser.add_argument("--target-altitude-m", type=float, default=2240.0)
    args = parser.parse_args()

    data = normalize_types(read_pitch_data(args.input))
    physics = add_physics_features(
        data,
        reference_altitude_m=args.reference_altitude_m,
        target_altitude_m=args.target_altitude_m,
    )
    keep = [
        "PitchUID", "year", "game_anon_id", "pitcher_anon_id", "PitcherThrows", "AutoPitchType",
        "altitude_category", "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension", "InducedVertBreak",
        "HorzBreak", "spin_axis_sin", "spin_axis_cos", "spin_parameter",
        "harp_air_density_ratio", "harp_magnus_scale", "aero_longitudinal_accel_ft_s2",
        "aero_transverse_x_accel_ft_s2", "aero_transverse_y_accel_ft_s2",
        "aero_transverse_z_accel_ft_s2", "aero_transverse_magnitude_ft_s2",
    ]
    keep = [column for column in keep if column in physics]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    physics[keep].to_parquet(output, index=False)
    print(f"Guardadas {len(physics):,} filas en {output}")


if __name__ == "__main__":
    main()
