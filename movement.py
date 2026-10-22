"""Modelos de movimiento explicados por spin/shape, con validación por pitcher."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MOVEMENT_TARGETS = {"ivb": "InducedVertBreak", "hb": "HorzBreak"}
MOVEMENT_NUMERICAL = ["RelSpeed", "SpinRate", "RelHeight", "RelSide", "Extension"]
MOVEMENT_CATEGORICAL = ["AutoPitchType", "PitcherThrows", "altitude_category"]


@dataclass
class FittedMovementModels:
    models: dict[str, Pipeline]
    numerical: list[str]
    categorical: list[str]


def _pipeline(numerical: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="median"), numerical),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.3,
    )
    estimator = LGBMRegressor(
        objective="regression_l2",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _usable_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
    required = ["pitcher_anon_id", target, *MOVEMENT_NUMERICAL, *MOVEMENT_CATEGORICAL]
    return data.dropna(subset=["pitcher_anon_id", target]).copy()


def train_movement_models(
    data: pd.DataFrame, n_splits: int = 5
) -> tuple[FittedMovementModels, pd.DataFrame, dict[str, dict[str, float]]]:
    """Genera predicciones OOF y reentrena modelos finales en toda la muestra.

    OOF significa out-of-fold: el pitcher evaluado no aparece en el entrenamiento
    de esa predicción. Por eso el residual resultante es menos optimista.
    """
    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}
    oof = data[["PitchUID", "year", "pitcher_anon_id", "AutoPitchType", "altitude_category"]].copy()

    for short_name, target in MOVEMENT_TARGETS.items():
        usable = _usable_data(data, target)
        unique_pitchers = usable["pitcher_anon_id"].nunique()
        if unique_pitchers < n_splits:
            raise ValueError(f"Solo hay {unique_pitchers} pitchers para {n_splits} folds.")
        features = usable[MOVEMENT_NUMERICAL + MOVEMENT_CATEGORICAL]
        predictions = pd.Series(np.nan, index=usable.index, dtype=float)
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(features, groups=usable["pitcher_anon_id"]):
            model = _pipeline(MOVEMENT_NUMERICAL, MOVEMENT_CATEGORICAL)
            model.fit(features.iloc[train_idx], usable[target].iloc[train_idx])
            predictions.iloc[test_idx] = model.predict(features.iloc[test_idx])

        observed = usable[target]
        residual = observed - predictions
        oof[f"{short_name}_observed"] = observed.reindex(data.index)
        oof[f"{short_name}_spin_shape_pred_oof"] = predictions.reindex(data.index)
        oof[f"{short_name}_non_magnus_residual_proxy"] = residual.reindex(data.index)
        metrics[short_name] = {
            "mae_inches": float(mean_absolute_error(observed, predictions)),
            "rmse_inches": float(np.sqrt(mean_squared_error(observed, predictions))),
            "r2": float(r2_score(observed, predictions)),
            "rows": int(len(usable)),
        }

        final_model = _pipeline(MOVEMENT_NUMERICAL, MOVEMENT_CATEGORICAL)
        final_model.fit(features, observed)
        fitted[short_name] = final_model

    return FittedMovementModels(fitted, MOVEMENT_NUMERICAL, MOVEMENT_CATEGORICAL), oof, metrics


def predict_movement(models: FittedMovementModels, data: pd.DataFrame) -> pd.DataFrame:
    """Predice el componente promedio explicado por spin/shape."""
    features = data[models.numerical + models.categorical]
    return pd.DataFrame(
        {
            f"{short_name}_spin_shape_pred": model.predict(features)
            for short_name, model in models.models.items()
        },
        index=data.index,
    )


def residual_profiles(oof: pd.DataFrame, shrinkage_k: float = 100.0) -> pd.DataFrame:
    """Promedia y encoge residuales por pitcher-temporada-tipo de pitch."""
    keys = ["year", "pitcher_anon_id", "AutoPitchType"]
    profiles = (
        oof.groupby(keys, as_index=False, observed=True)
        .agg(
            n_pitches=("PitchUID", "size"),
            ivb_residual_mean=("ivb_non_magnus_residual_proxy", "mean"),
            hb_residual_mean=("hb_non_magnus_residual_proxy", "mean"),
        )
    )
    weight = profiles["n_pitches"] / (profiles["n_pitches"] + shrinkage_k)
    profiles["ivb_non_magnus_residual_shrunk"] = weight * profiles["ivb_residual_mean"]
    profiles["hb_non_magnus_residual_shrunk"] = weight * profiles["hb_residual_mean"]
    return profiles
    


