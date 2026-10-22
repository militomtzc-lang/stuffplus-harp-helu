"""Genera gráficas de altitud, movimiento explicado y proxy no-Magnus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PITCH_TYPES = ["Four-Seam", "Sinker", "Slider", "Curveball", "Changeup"]
ALTITUDE_ORDER = ["No Altitude", "Medium Altitude", "Extreme Altitude"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-input", required=True)
    parser.add_argument("--oof-input", required=True)
    parser.add_argument("--profiles-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-pitches", type=int, default=50)
    args = parser.parse_args()

    physics = pd.read_parquet(args.physics_input)
    oof = pd.read_parquet(args.oof_input)
    data = physics[["PitchUID", "altitude_category", "AutoPitchType", "InducedVertBreak", "HorzBreak"]].merge(
        oof[
            [
                "PitchUID", "ivb_spin_shape_pred_oof", "hb_spin_shape_pred_oof",
                "ivb_non_magnus_residual_proxy", "hb_non_magnus_residual_proxy",
            ]
        ],
        on="PitchUID",
        how="inner",
    )
    data = data.loc[data["AutoPitchType"].isin(PITCH_TYPES)].dropna().copy()
    data["abs_hb"] = data["HorzBreak"].abs()
    summary = (
        data.groupby(["altitude_category", "AutoPitchType"], as_index=False, observed=True)
        .agg(
            pitches=("PitchUID", "size"),
            mean_ivb=("InducedVertBreak", "mean"),
            mean_abs_hb=("abs_hb", "mean"),
            mean_ivb_spin_shape=("ivb_spin_shape_pred_oof", "mean"),
            mean_hb_spin_shape=("hb_spin_shape_pred_oof", "mean"),
            mean_ivb_non_magnus_residual=("ivb_non_magnus_residual_proxy", "mean"),
            mean_hb_non_magnus_residual=("hb_non_magnus_residual_proxy", "mean"),
        )
    )
    summary["altitude_category"] = pd.Categorical(
        summary["altitude_category"], ALTITUDE_ORDER, ordered=True
    )
    summary = summary.sort_values(["AutoPitchType", "altitude_category"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "altitude_movement_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pitch_type in PITCH_TYPES:
        subset = summary.loc[summary["AutoPitchType"].eq(pitch_type)].set_index("altitude_category").reindex(ALTITUDE_ORDER)
        axes[0].plot(ALTITUDE_ORDER, subset["mean_ivb"], marker="o", label=pitch_type)
        axes[1].plot(ALTITUDE_ORDER, subset["mean_abs_hb"], marker="o", label=pitch_type)
    axes[0].axhline(0, color="0.75", linewidth=1)
    axes[0].set_title("IVB promedio por altitud")
    axes[0].set_ylabel("Induced vertical break (pulgadas)")
    axes[1].set_title("Movimiento horizontal absoluto por altitud")
    axes[1].set_ylabel("|Horizontal break| (pulgadas)")
    for axis in axes:
        axis.set_xlabel("Categoría de altitud")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(title="Tipo de pitch", loc="best")
    fig.suptitle("Movimiento observado: comparación descriptiva, no causal")
    fig.tight_layout()
    fig.savefig(output_dir / "altitude_movement_contrast.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    profiles = pd.read_csv(args.profiles_input)
    profiles = profiles.loc[
        profiles["AutoPitchType"].isin(PITCH_TYPES) & profiles["n_pitches"].ge(args.min_pitches)
    ].dropna(subset=["ivb_non_magnus_residual_shrunk", "hb_non_magnus_residual_shrunk"])
    fig, axis = plt.subplots(figsize=(8, 7))
    for pitch_type in PITCH_TYPES:
        subset = profiles.loc[profiles["AutoPitchType"].eq(pitch_type)]
        axis.scatter(
            subset["ivb_non_magnus_residual_shrunk"],
            subset["hb_non_magnus_residual_shrunk"],
            s=10 + subset["n_pitches"].clip(upper=300) / 8,
            alpha=0.35,
            label=pitch_type,
        )
    axis.axhline(0, color="0.55", linewidth=1)
    axis.axvline(0, color="0.55", linewidth=1)
    axis.set_title("Perfiles de residual no-Magnus por pitcher-temporada-pitch")
    axis.set_xlabel("Residual IVB ajustado (pulgadas)")
    axis.set_ylabel("Residual HB ajustado (pulgadas)")
    axis.legend(title="Tipo de pitch", loc="best")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "non_magnus_proxy_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráficas y tabla guardadas en {output_dir}")


if __name__ == "__main__":
    main()
