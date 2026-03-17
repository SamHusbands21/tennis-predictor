"""
Calibration helpers for the tennis betting model.

The Random Forest is wrapped in CalibratedClassifierCV (isotonic regression)
to ensure its output probabilities are well-calibrated. XGBoost's logistic
output is already reasonably calibrated for binary classification.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator


def isotonic_calibrate(
    estimator: BaseEstimator,
    X_train,
    y_train,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Wrap an already-fitted estimator in isotonic calibration trained on
    (X_train, y_train) via cross_val_predict to avoid using the same data
    for both fitting and calibration.
    """
    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated
