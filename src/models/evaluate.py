"""
Evaluation of model performance on the held-out test set.

Classification metrics:
  - Log loss (primary — penalises miscalibration)
  - Brier score
  - ROC-AUC (binary)
  - Accuracy
  - Precision and recall
  - Calibration reliability diagram (saved as PNG)

Profitability metrics (using Betfair odds stored during feature engineering,
if available — odds_p1 / odds_p2 columns):
  - Value bets identified when model_prob × decimal_odds > EV_THRESHOLD
  - Flat staking: 1 unit per bet, ROI and total P&L
  - Kelly staking: fraction = (p × odds − 1) / (odds − 1), capped at MAX_KELLY
  - Sharpe ratio of bet-level returns

Feature importance:
  - SHAP TreeExplainer on XGBoost model (saved as PNG)

All results saved to:
  output/evaluation_report.json   — full detailed report
  output/website_evaluation.json  — simplified schema for tennis-paper.html
  output/calibration.png
  output/shap_summary.png
  output/pnl_curve.png
  output/threshold_sweep.png
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.features.engineer import MODEL_FEATURES
from src.models.train import TEST_START_YEAR, _load_features, _split, ensemble_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path(__file__).parents[2] / "output"
EV_THRESHOLD = 1.20         # bet when model_prob × odds > this
MAX_KELLY    = 0.25         # cap Kelly fraction at 25% of bankroll

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "elo_p1":                    "Player1 overall Elo",
    "elo_p2":                    "Player2 overall Elo",
    "elo_diff":                  "Elo difference (P1 − P2)",
    "surface_elo_p1":            "Player1 surface Elo",
    "surface_elo_p2":            "Player2 surface Elo",
    "surface_elo_diff":          "Surface Elo diff (P1 − P2)",
    "p1_win_rate_5":             "P1 win rate (last 5)",
    "p2_win_rate_5":             "P2 win rate (last 5)",
    "p1_win_rate_10":            "P1 win rate (last 10)",
    "p2_win_rate_10":            "P2 win rate (last 10)",
    "p1_surface_win_rate_10":    "P1 surface win rate (last 10)",
    "p2_surface_win_rate_10":    "P2 surface win rate (last 10)",
    "p1_first_serve_pct":        "P1 1st serve % (last 10)",
    "p2_first_serve_pct":        "P2 1st serve % (last 10)",
    "p1_first_serve_won_pct":    "P1 1st serve pts won % (last 10)",
    "p2_first_serve_won_pct":    "P2 1st serve pts won % (last 10)",
    "p1_bp_save_rate":           "P1 break-point save rate (last 10)",
    "p2_bp_save_rate":           "P2 break-point save rate (last 10)",
    "p1_days_rest":              "P1 days since last match",
    "p2_days_rest":              "P2 days since last match",
    "p1_prev_match_minutes":     "P1 previous match duration (min)",
    "p2_prev_match_minutes":     "P2 previous match duration (min)",
    "rank_diff":                 "Rank difference (P2 rank − P1 rank)",
    "p1_rank_pts":               "P1 ATP ranking points",
    "p2_rank_pts":               "P2 ATP ranking points",
    "h2h_p1_win_rate":           "H2H win rate (P1, last 10 meetings)",
    "tourney_level_encoded":     "Tournament level (GS=4, M=3, …)",
    "best_of":                   "Best of (3 or 5)",
    "surface_hard":              "Surface: Hard",
    "surface_clay":              "Surface: Clay",
    "surface_grass":             "Surface: Grass",
}

DISPLAY_NAMES = [FEATURE_DISPLAY_NAMES.get(f, f) for f in MODEL_FEATURES]


# ── Classification helpers ────────────────────────────────────────────────────

def _brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Binary Brier score: mean squared error of probability estimates."""
    return float(np.mean((proba - y_true.astype(float)) ** 2))


def _calibration_diagram(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> None:
    """Save a reliability (calibration) diagram for binary predictions."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Calibration Reliability Diagram (Test Set)", fontsize=12)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_fracs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 0:
            bin_means.append(proba[mask].mean())
            bin_fracs.append(y_true[mask].mean())

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(bin_means, bin_fracs, "o-", color="#6cabdd", label="Model")
    ax.set_xlabel("Mean predicted probability (P1 wins)")
    ax.set_ylabel("Fraction of P1 wins")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = save_path or OUTPUT_DIR / "calibration.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Calibration diagram saved to {path}")


def _shap_summary_plot(
    xgb_model,
    X_test: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """SHAP feature importance bar chart for the XGBoost model."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping. Run: pip install shap")
        return

    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # For binary XGBoost, shap_values may be 2D (n_samples, n_features)
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    order = np.argsort(mean_abs)
    ordered_names = [DISPLAY_NAMES[i] for i in order]
    ordered_vals  = mean_abs[order]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(ordered_names))
    ax.barh(y_pos, ordered_vals, color="#6cabdd", edgecolor="none", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title("Feature Importance (XGBoost — SHAP)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()

    path = save_path or OUTPUT_DIR / "shap_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"SHAP summary plot saved to {path}")


# ── Profitability helpers ──────────────────────────────────────────────────────

def _value_bets(
    proba: np.ndarray,
    odds_p1: np.ndarray,
    odds_p2: np.ndarray,
    y_true: np.ndarray,
    threshold: float = EV_THRESHOLD,
) -> pd.DataFrame:
    """Identify value bets and compute flat / Kelly returns per bet."""
    rows = []
    for i in range(len(y_true)):
        for outcome, p, o, correct_cls in [
            ("p1", proba[i],       odds_p1[i], 1),
            ("p2", 1 - proba[i],   odds_p2[i], 0),
        ]:
            if np.isnan(o) or o <= 1.0:
                continue
            ev = p * o
            if ev < threshold:
                continue
            kelly = min(MAX_KELLY, (p * o - 1) / (o - 1))
            won = int(y_true[i] == correct_cls)
            flat_return   = o - 1 if won else -1.0
            kelly_return  = kelly * (o - 1) if won else -kelly

            rows.append({
                "match_idx":      i,
                "outcome":        outcome,
                "p_model":        round(p, 4),
                "odds":           round(o, 2),
                "ev":             round(ev, 4),
                "kelly_fraction": round(kelly, 4),
                "won":            won,
                "flat_return":    round(flat_return, 4),
                "kelly_return":   round(kelly_return, 4),
            })
    return pd.DataFrame(rows)


def _profitability_summary(bets: pd.DataFrame, label: str) -> dict:
    if len(bets) == 0:
        return {"label": label, "n_bets": 0}

    flat_returns   = bets["flat_return"].values
    kelly_returns  = bets["kelly_return"].values
    n = len(bets)

    flat_roi     = float(flat_returns.sum() / n * 100)
    kelly_pnl    = float(kelly_returns.sum())
    kelly_sharpe = float(kelly_returns.mean() / (kelly_returns.std() + 1e-9) * np.sqrt(n))

    return {
        "label":           label,
        "n_bets":          n,
        "win_rate_pct":    round(float(bets["won"].mean() * 100), 1),
        "flat_roi_pct":    round(flat_roi, 2),
        "flat_total_pnl":  round(float(flat_returns.sum()), 2),
        "kelly_total_pnl": round(kelly_pnl, 4),
        "kelly_sharpe":    round(kelly_sharpe, 3),
    }


def _naive_baseline_roi(
    odds_p1: np.ndarray,
    odds_p2: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Always bet on the favourite (lower odds). Flat staking."""
    odds_arr = np.stack([odds_p1, odds_p2], axis=1)
    valid = ~np.any(np.isnan(odds_arr), axis=1)
    odds_v = odds_arr[valid]
    y_v    = y_true[valid]
    fav_cls = np.argmin(odds_v, axis=1)  # 0=p1 favourite, 1=p2 favourite
    won = (fav_cls == (1 - y_v)).astype(float)  # p1 wins when target=1 = cls 0
    # Actually: fav_cls=0 means p1 is favourite; p1 wins if y_v=1
    won = ((fav_cls == 0) == (y_v == 1)).astype(float)
    returns = np.where(won, odds_v[np.arange(len(odds_v)), fav_cls] - 1, -1.0)
    return {
        "label":          "Always-favourite baseline",
        "n_bets":         int(len(returns)),
        "win_rate_pct":   round(float(won.mean() * 100), 1),
        "flat_roi_pct":   round(float(returns.mean() * 100), 2),
        "flat_total_pnl": round(float(returns.sum()), 2),
    }


def _threshold_sweep(
    proba: np.ndarray,
    odds_p1: np.ndarray,
    odds_p2: np.ndarray,
    y_true: np.ndarray,
    thresholds: Optional[list] = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = [round(1.00 + i * 0.05, 2) for i in range(1, 11)]

    all_bets = _value_bets(proba, odds_p1, odds_p2, y_true, threshold=1.0)
    rows = []
    for t in thresholds:
        bets = all_bets[all_bets["ev"] >= t]
        if len(bets) == 0:
            rows.append({"ev_threshold": t, "n_bets": 0,
                         "flat_roi_pct": None, "flat_pnl": None,
                         "win_rate_pct": None, "kelly_sharpe": None})
            continue
        flat_returns  = bets["flat_return"].values
        kelly_returns = bets["kelly_return"].values
        n = len(bets)
        kelly_sharpe = float(
            kelly_returns.mean() / (kelly_returns.std() + 1e-9) * np.sqrt(n)
        )
        rows.append({
            "ev_threshold":  t,
            "n_bets":        n,
            "flat_roi_pct":  round(float(flat_returns.sum() / n * 100), 2),
            "flat_pnl":      round(float(flat_returns.sum()), 2),
            "win_rate_pct":  round(float(bets["won"].mean() * 100), 1),
            "kelly_sharpe":  round(kelly_sharpe, 3),
        })
    return rows


def _sweep_plot(rows: list[dict], save_path: Path) -> None:
    thresholds = [r["ev_threshold"] for r in rows]

    def _vals(key):
        return [r.get(key) for r in rows]

    roi = _vals("flat_roi_pct")
    n   = _vals("n_bets")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"EV Threshold Sweep — Test Set ({TEST_START_YEAR}–present)",
        fontsize=12, fontweight="bold",
    )

    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.plot(thresholds, roi, "o-", color="#6cabdd", linewidth=2, markersize=6)
    ax1.set_xlabel("EV threshold", fontsize=9)
    ax1.set_ylabel("Flat ROI (%)", fontsize=9)
    ax1.set_title("ROI vs EV threshold", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=8)

    ax2.plot(thresholds, n, "o-", color="#e07b54", linewidth=2, markersize=6)
    ax2.set_xlabel("EV threshold", fontsize=9)
    ax2.set_ylabel("Number of bets", fontsize=9)
    ax2.set_title("Bet volume vs EV threshold", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Threshold sweep plot saved to {save_path}")


def _pnl_curve_plot(
    bets: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Path,
) -> None:
    if len(bets) == 0:
        logger.warning("No bets for P&L curve — skipping plot.")
        return

    idx_to_date = test_df["date"].reset_index(drop=True)
    bets = bets.copy()
    bets["date"] = bets["match_idx"].map(idx_to_date)
    bets = bets.sort_values("date").reset_index(drop=True)
    bets["cum_pnl"] = bets["flat_return"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    x = bets.index.values
    y = bets["cum_pnl"].values
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.15, color="#16a34a", interpolate=True)
    ax.fill_between(x, y, 0, where=(y < 0),  alpha=0.15, color="#dc2626", interpolate=True)
    ax.plot(x, y, color="#6cabdd", linewidth=2)

    final = round(float(y[-1]), 2)
    color = "#16a34a" if final >= 0 else "#dc2626"
    sign  = "+" if final >= 0 else ""
    ax.annotate(
        f"Final: {sign}{final} units",
        xy=(x[-1], y[-1]),
        xytext=(-10, 12),
        textcoords="offset points",
        fontsize=9, color=color, fontweight="bold",
    )

    ax.set_xlabel("Bet number (chronological)", fontsize=9)
    ax.set_ylabel("Cumulative flat P&L (units)", fontsize=9)
    ax.set_title(
        f"Cumulative P&L — EV > {EV_THRESHOLD} (test set {TEST_START_YEAR}–present)",
        fontsize=10, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"P&L curve saved to {save_path}")


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(xgb_model, rf_model, test_df: pd.DataFrame) -> dict:
    """Full evaluation on the test set. Returns a results dict."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = test_df.sort_values("date").reset_index(drop=True)
    X_test = test_df[MODEL_FEATURES].fillna(0).values
    y_test = test_df["target"].values.astype(int)

    proba = ensemble_proba(xgb_model, rf_model, X_test)

    # ── Classification metrics ────────────────────────────────────────────────
    pred_cls = (proba >= 0.5).astype(int)
    ll       = log_loss(y_test, np.stack([1 - proba, proba], axis=1))
    brier    = _brier_score(y_test, proba)
    roc_auc  = roc_auc_score(y_test, proba)
    acc      = accuracy_score(y_test, pred_cls)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, pred_cls, average=None, labels=[0, 1], zero_division=0
    )

    logger.info(f"\n=== Classification (test set: {len(test_df)} matches) ===")
    logger.info(f"  Log loss:    {ll:.4f}")
    logger.info(f"  Brier score: {brier:.4f}")
    logger.info(f"  ROC-AUC:     {roc_auc:.4f}")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Precision — P2: {precision[0]:.3f}  P1: {precision[1]:.3f}")
    logger.info(f"  Recall    — P2: {recall[0]:.3f}  P1: {recall[1]:.3f}")

    _calibration_diagram(y_test, proba, save_path=OUTPUT_DIR / "calibration.png")
    _shap_summary_plot(xgb_model, X_test, save_path=OUTPUT_DIR / "shap_summary.png")

    # ── Profitability (if historical odds are available) ──────────────────────
    odds_p1 = test_df["odds_p1"].values.astype(float)
    odds_p2 = test_df["odds_p2"].values.astype(float)

    has_odds = not (np.isnan(odds_p1).all() and np.isnan(odds_p2).all())
    model_summary = {"label": "Model value bets", "n_bets": 0}
    baseline_summary = {"label": "Always-favourite baseline", "n_bets": 0}
    sweep_rows = []
    model_bets = pd.DataFrame()

    if has_odds:
        logger.info("\n=== Profitability analysis ===")
        model_bets = _value_bets(proba, odds_p1, odds_p2, y_test, threshold=EV_THRESHOLD)
        model_summary = _profitability_summary(model_bets, "Model value bets")
        baseline_summary = _naive_baseline_roi(odds_p1, odds_p2, y_test)
        sweep_rows = _threshold_sweep(proba, odds_p1, odds_p2, y_test)

        logger.info(f"  Value bets (EV > {EV_THRESHOLD}): {model_summary.get('n_bets', 0)}")
        logger.info(f"  Flat ROI: {model_summary.get('flat_roi_pct', 'N/A')}%")
        logger.info(f"  Kelly Sharpe: {model_summary.get('kelly_sharpe', 'N/A')}")

        _pnl_curve_plot(model_bets, test_df, OUTPUT_DIR / "pnl_curve.png")
        _sweep_plot(sweep_rows, OUTPUT_DIR / "threshold_sweep.png")
        with open(OUTPUT_DIR / "threshold_sweep.json", "w") as f:
            json.dump(sweep_rows, f, indent=2)
    else:
        logger.info("No historical odds available — skipping profitability analysis.")

    # ── Test set year range ───────────────────────────────────────────────────
    years = sorted(test_df["year"].dropna().unique().tolist()) if "year" in test_df.columns else []
    year_range = f"{int(min(years))}–{int(max(years))}" if years else f"{TEST_START_YEAR}–present"

    report = {
        "test_set": {
            "n_matches":  len(test_df),
            "year_range": year_range,
        },
        "classification": {
            "log_loss":   round(float(ll), 4),
            "brier_score":round(float(brier), 4),
            "roc_auc":    round(float(roc_auc), 4),
            "accuracy":   round(float(acc), 4),
            "precision":  {
                "p1": round(float(precision[1]), 4),
                "p2": round(float(precision[0]), 4),
            },
            "recall": {
                "p1": round(float(recall[1]), 4),
                "p2": round(float(recall[0]), 4),
            },
        },
        "profitability": {
            "has_odds":      has_odds,
            "ev_threshold":  EV_THRESHOLD,
            "model":         model_summary,
            "baseline":      baseline_summary,
            "threshold_sweep": sweep_rows,
        },
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {report_path}")

    _write_website_json(report, year_range)
    return report


def _write_website_json(report: dict, year_range: str) -> None:
    """
    Write a simplified evaluation JSON consumed by tennis-paper.html.
    Saved to output/website_evaluation.json; the retrain script copies it to
    the website's data/ folder as data/tennis_evaluation.json.
    """
    clf  = report["classification"]
    prof = report["profitability"]
    m    = prof.get("model", {})

    website_json = {
        "_note": "Generated by src/models/evaluate.py. Do not edit manually.",
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "test_year_range": year_range,
        "n_test_matches":  report["test_set"]["n_matches"],
        "roc_auc":         clf.get("roc_auc"),
        "brier_score":     clf.get("brier_score"),
        "log_loss":        clf.get("log_loss"),
        "accuracy":        clf.get("accuracy"),
        "precision": clf.get("precision", {"p1": None, "p2": None}),
        "recall":    clf.get("recall",    {"p1": None, "p2": None}),
        "strategy": {
            "ev_threshold":    prof.get("ev_threshold"),
            "has_odds":        prof.get("has_odds", False),
        },
        "gambling": {
            "value_bet_threshold": prof.get("ev_threshold"),
            "flat_roi_pct":        m.get("flat_roi_pct"),
            "flat_bets":           m.get("n_bets"),
            "flat_pnl":            m.get("flat_total_pnl"),
            "kelly_pnl":           m.get("kelly_total_pnl"),
            "sharpe_ratio":        m.get("kelly_sharpe"),
        },
    }

    path = OUTPUT_DIR / "website_evaluation.json"
    with open(path, "w") as f:
        json.dump(website_json, f, indent=2)
    logger.info(f"Website evaluation JSON saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.models.train import load_models

    xgb_model, rf_model = load_models()
    df = _load_features()
    _, test_df = _split(df)
    run_evaluation(xgb_model, rf_model, test_df)
