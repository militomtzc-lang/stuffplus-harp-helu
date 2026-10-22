"""Entrenamiento, validación por pitcher y escala Stuff+."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGETS = ["target_whiff", "target_called_strike", "target_groundball", "target_hard_contact"]
WEIGHTS = {
    "target_whiff": 0.45,
    "target_called_strike": 0.15,
    "target_groundball": 0.15,
    "target_hard_contact": -0.25,
}


@dataclass
class FittedStuffModel:
    models: dict[str, Pipeline]
    numerical: list[str]
    categorical: list[str]
    raw_mean: float
    raw_std: float


def _pipeline(numerical: list[str], categorical: list[str], scale_pos_weight: float) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="median"), numerical),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        sparse_threshold=0.3,
    )
    estimator = LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _raw_quality(predictions: pd.DataFrame) -> pd.Series:
    return sum(WEIGHTS[target] * predictions[f"p_{target.removeprefix('target_')}"] for target in TARGETS)


def _fit_models(
    data: pd.DataFrame, numerical: list[str], categorical: list[str]
) -> dict[str, Pipeline]:
    features = data[numerical + categorical]
    fitted: dict[str, Pipeline] = {}
    for target in TARGETS:
        y = data[target].astype(int)
        positives = y.sum()
        negatives = len(y) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"El target {target} no tiene ambas clases en entrenamiento.")
        model = _pipeline(numerical, categorical, scale_pos_weight=negatives / positives)
        model.fit(features, y)
        fitted[target] = model
    return fitted


def predict_probabilities(model: FittedStuffModel | dict[str, Pipeline], data: pd.DataFrame) -> pd.DataFrame:
    models = model.models if isinstance(model, FittedStuffModel) else model
    numerical = model.numerical if isinstance(model, FittedStuffModel) else None
    categorical = model.categorical if isinstance(model, FittedStuffModel) else None
    features = data[numerical + categorical] if numerical is not None else data
    output = pd.DataFrame(index=data.index)
    for target, fitted in models.items():
        output[f"p_{target.removeprefix('target_')}"] = fitted.predict_proba(features)[:, 1]
    return output


def fit_validate_and_score(
    data: pd.DataFrame, numerical: list[str], categorical: list[str]
) -> tuple[FittedStuffModel, pd.DataFrame, dict[str, dict[str, float]]]:
    """Hace holdout de pitchers, reporta métricas y después reentrena en todo."""
    usable = data.dropna(subset=["pitcher_anon_id"]).copy()
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(split.split(usable, groups=usable["pitcher_anon_id"]))
    train, test = usable.iloc[train_idx], usable.iloc[test_idx]
    holdout_models = _fit_models(train, numerical, categorical)
    holdout_probabilities = predict_probabilities(holdout_models, test[numerical + categorical])
    metrics: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        column = f"p_{target.removeprefix('target_')}"
        metrics[target] = {
            "roc_auc": float(roc_auc_score(test[target], holdout_probabilities[column])),
            "brier": float(brier_score_loss(test[target], holdout_probabilities[column])),
            "prevalence": float(test[target].mean()),
        }

    final_models = _fit_models(usable, numerical, categorical)
    final = FittedStuffModel(
        models=final_models,
        numerical=numerical,
        categorical=categorical,
        raw_mean=0.0,
        raw_std=1.0,
    )
    reference_probabilities = predict_probabilities(final, usable)
    raw_reference = _raw_quality(reference_probabilities)
    final.raw_mean = float(raw_reference.mean())
    final.raw_std = float(raw_reference.std(ddof=0))
    if final.raw_std == 0:
        raise ValueError("La desviación del score bruto es cero.")

    holdout = test[["PitchUID", "year", "pitcher_anon_id", "AutoPitchType"]].copy()
    holdout = holdout.join(holdout_probabilities)
    holdout["quality_raw"] = _raw_quality(holdout_probabilities)
    holdout["stuff_plus"] = 100 + 10 * (holdout["quality_raw"] - final.raw_mean) / final.raw_std
    return final, holdout, metrics


def score_all(model: FittedStuffModel, data: pd.DataFrame) -> pd.DataFrame:
    probabilities = predict_probabilities(model, data)
    scored = data[["PitchUID", "year", "pitcher_anon_id", "AutoPitchType"]].copy().join(probabilities)
    scored["quality_raw"] = _raw_quality(probabilities)
    scored["stuff_plus"] = 100 + 10 * (scored["quality_raw"] - model.raw_mean) / model.raw_std
    return scored


def altitude_counterfactual(model: FittedStuffModel, data: pd.DataFrame) -> pd.Series:
    """Delta asociativo Extreme Altitude menos No Altitude; no es efecto causal."""
    if "altitude_category" not in model.categorical:
        return pd.Series(np.nan, index=data.index, name="extreme_minus_no_altitude_stuff_plus")
    high = data.copy()
    low = data.copy()
    high["altitude_category"] = "Extreme Altitude"
    low["altitude_category"] = "No Altitude"
    high_raw = _raw_quality(predict_probabilities(model, high))
    low_raw = _raw_quality(predict_probabilities(model, low))
    return 10 * (high_raw - low_raw) / model.raw_std
