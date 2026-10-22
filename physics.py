"""Capa física transparente para el simulador de altitud.

No intenta identificar seam-shifted wake (SSW) de manera definitiva: el
dataset no tiene orientación de costuras, spin axis 3D ni spin efficiency.
En cambio prepara variables físicas y una forma reproducible de estimar un
residual no-Magnus con un modelo estadístico posterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BALL_RADIUS_M = 0.0367
SEA_LEVEL_DENSITY_KG_M3 = 1.225
GRAVITY_FT_S2 = 32.174
MPH_TO_MPS = 0.44704


def air_density_isa(altitude_m: float | np.ndarray) -> float | np.ndarray:
    """Densidad del aire bajo atmósfera estándar internacional, troposfera.

    Es una referencia para sensibilidad física, no un reemplazo de temperatura,
    humedad y presión observadas en cada juego.
    """
    altitude = np.asarray(altitude_m, dtype=float)
    temperature_k = 288.15 - 0.0065 * altitude
    pressure_pa = 101325 * (temperature_k / 288.15) ** 5.25588
    density = pressure_pa / (287.05 * temperature_k)
    return float(density) if density.ndim == 0 else density


def density_ratio(target_altitude_m: float, reference_altitude_m: float = 0.0) -> float:
    """Proporción rho_target/rho_reference para transportar fuerza aerodinámica."""
    return float(air_density_isa(target_altitude_m) / air_density_isa(reference_altitude_m))


def add_physics_features(
    data: pd.DataFrame,
    reference_altitude_m: float = 0.0,
    target_altitude_m: float = 2240.0,
) -> pd.DataFrame:
    """Crea features físicas sin outcomes ni ubicación de plato.

    La aceleración aerodinámica se estima restando gravedad y proyectando la
    aceleración restante sobre la velocidad. El componente perpendicular es
    movimiento aerodinámico total: contiene Magnus y potencialmente SSW.
    """
    output = data.copy()
    speed_mps = pd.to_numeric(output["RelSpeed"], errors="coerce") * MPH_TO_MPS
    rpm = pd.to_numeric(output["SpinRate"], errors="coerce")
    axis_rad = np.deg2rad(pd.to_numeric(output["SpinAxis"], errors="coerce"))
    omega_rad_s = rpm * (2 * np.pi / 60)

    output["spin_axis_sin"] = np.sin(axis_rad)
    output["spin_axis_cos"] = np.cos(axis_rad)
    output["spin_parameter"] = BALL_RADIUS_M * omega_rad_s / speed_mps
    output["harp_air_density_ratio"] = density_ratio(target_altitude_m, reference_altitude_m)
    output["harp_magnus_scale"] = output["harp_air_density_ratio"]

    trajectory_columns = ["vx0", "vy0", "vz0", "ax0", "ay0", "az0"]
    if not set(trajectory_columns).issubset(output.columns):
        return output

    velocity = output[["vx0", "vy0", "vz0"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    acceleration = output[["ax0", "ay0", "az0"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    speed = np.linalg.norm(velocity, axis=1)
    valid = np.isfinite(speed) & (speed > 0) & np.isfinite(acceleration).all(axis=1)
    unit_velocity = np.full_like(velocity, np.nan)
    unit_velocity[valid] = velocity[valid] / speed[valid, None]

    gravity = np.array([0.0, 0.0, -GRAVITY_FT_S2])
    aero_acceleration = acceleration - gravity
    longitudinal = np.sum(aero_acceleration * unit_velocity, axis=1)
    drag_vector = longitudinal[:, None] * unit_velocity
    transverse = aero_acceleration - drag_vector

    output["aero_longitudinal_accel_ft_s2"] = longitudinal
    output["aero_transverse_x_accel_ft_s2"] = transverse[:, 0]
    output["aero_transverse_y_accel_ft_s2"] = transverse[:, 1]
    output["aero_transverse_z_accel_ft_s2"] = transverse[:, 2]
    output["aero_transverse_magnitude_ft_s2"] = np.linalg.norm(transverse, axis=1)
    return output


def simulate_density_scaled_movement(
    magnus_like_ivb: pd.Series | np.ndarray,
    magnus_like_hb: pd.Series | np.ndarray,
    non_magnus_ivb: pd.Series | np.ndarray = 0.0,
    non_magnus_hb: pd.Series | np.ndarray = 0.0,
    target_altitude_m: float = 2240.0,
    reference_altitude_m: float = 0.0,
    ssw_density_scale: float = 1.0,
) -> pd.DataFrame:
    """Transporta movimiento a otra densidad como análisis de sensibilidad.

    `ssw_density_scale` permite probar supuestos: 0 conserva el residual
    no-Magnus, 1 lo escala proporcionalmente a densidad. No equivale a medir
    la orientación de costuras.
    """
    ratio = density_ratio(target_altitude_m, reference_altitude_m)
    magnus_ivb = np.asarray(magnus_like_ivb, dtype=float)
    magnus_hb = np.asarray(magnus_like_hb, dtype=float)
    residual_ivb = np.asarray(non_magnus_ivb, dtype=float)
    residual_hb = np.asarray(non_magnus_hb, dtype=float)
    residual_ratio = 1 + ssw_density_scale * (ratio - 1)
    return pd.DataFrame(
        {
            "air_density_ratio": ratio,
            "simulated_ivb": ratio * magnus_ivb + residual_ratio * residual_ivb,
            "simulated_hb": ratio * magnus_hb + residual_ratio * residual_hb,
        }
    )
