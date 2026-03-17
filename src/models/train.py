"""
Train XGBoost and Random Forest classifiers for tennis match prediction.

Binary classification: target = 1 if player1 wins, 0 if player2 wins.
Player1/player2 assignment is randomised during feature engineering so the
model sees a balanced 50/50 split.

Train / test split (strict time-based holdout — NO leakage):
  Train + CV:  years 2005 – 2020  (16 seasons)
  Test:        years 2021 – present  (held out until final evaluation)

Walk-forward CV uses sklearn TimeSeriesSplit on the training set to tune
hyperparameters without any future leakage. Probabilities from the Random
Forest are calibrated with isotonic regression.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.features.engineer import MODEL_FEATURES

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR = Path(__file__).parents[2] / "models"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

TRAIN_CUTOFF_YEAR = 2020   # inclusive
TEST_START_YEAR   = 2021   # inclusive

N_CV_SPLITS = 5


def _load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/features/engineer.py first."
        )
    return pd.read_parquet(path)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on calendar year into train and test sets."""
    df = df.copy()
    year_col = pd.to_datetime(df["date"]).dt.year if "year" not in df.columns else df["year"]
    train = df[year_col <= TRAIN_CUTOFF_YEAR].copy()
    test  = df[year_col >= TEST_START_YEAR].copy()
    logger.info(
        f"Train: {len(train)} matches ({year_col[year_col <= TRAIN_CUTOFF_YEAR].min()}"
        f"–{TRAIN_CUTOFF_YEAR}) | "
        f"Test: {len(test)} matches ({TEST_START_YEAR}–present)"
    )
    return train, test


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Tune XGBoost via walk-forward grid search, then refit on full train set."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    param_grid = {
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.05, 0.1],
        "n_estimators":     [200, 400],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Tuning XGBoost with walk-forward CV...")
    gs = GridSearchCV(
        base, param_grid, scoring="neg_log_loss",
        cv=tscv, n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(f"  Best XGB params: {gs.best_params_}")
    logger.info(f"  Best CV log-loss: {-gs.best_score_:.4f}")
    return gs.best_estimator_


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    """Tune Random Forest, then wrap with isotonic calibration."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    param_grid = {
        "n_estimators":    [300, 500],
        "max_depth":       [None, 10, 15],
        "min_samples_leaf":[5, 10],
        "max_features":    ["sqrt", 0.5],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)

    logger.info("Tuning Random Forest with walk-forward CV...")
    gs = GridSearchCV(
        base, param_grid, scoring="neg_log_loss",
        cv=tscv, n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(f"  Best RF params: {gs.best_params_}")
    logger.info(f"  Best CV log-loss: {-gs.best_score_:.4f}")

    calibrated = CalibratedClassifierCV(
        gs.best_estimator_, method="isotonic", cv=5
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def ensemble_proba(
    xgb_model: XGBClassifier,
    rf_model: CalibratedClassifierCV,
    X: np.ndarray,
) -> np.ndarray:
    """
    Average calibrated probabilities from both models.

    Returns a 1-D array of P(player1 wins) for each row.
    """
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_rf  = rf_model.predict_proba(X)[:, 1]
    return (p_xgb + p_rf) / 2.0


def run_training() -> dict:
    """Full training pipeline. Returns dict with models and split DataFrames."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_features()
    train_df, test_df = _split(df)

    train_df = train_df.sort_values("date").reset_index(drop=True)

    X_train = train_df[MODEL_FEATURES].fillna(0).values
    y_train = train_df["target"].values

    xgb_model = train_xgboost(X_train, y_train)
    rf_model  = train_rf(X_train, y_train)

    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(rf_model,  MODELS_DIR / "rf_model.joblib")
    logger.info("Models saved to models/")

    return {
        "xgb_model": xgb_model,
        "rf_model":  rf_model,
        "train_df":  train_df,
        "test_df":   test_df,
    }


def load_models() -> tuple:
    """Load saved XGBoost and Random Forest models from disk."""
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.joblib")
    rf_model  = joblib.load(MODELS_DIR / "rf_model.joblib")
    return xgb_model, rf_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_training()
    logger.info("Training complete.")
