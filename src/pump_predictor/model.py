"""Train and evaluate the pump classifier.

Design choices, all standard for rare-event financial prediction:
  * Time-based split — the test set is the most recent dates; random splits
    leak overlapping forward windows and inflate scores.
  * Purge gap — `forward_window` days dropped between train and test so no
    training label's forward window overlaps test features.
  * Class imbalance — pumps are ~1-5% of rows; LightGBM `scale_pos_weight`,
    and PR-AUC / precision@k as the metrics that matter. ROC-AUC reported but
    over-optimistic under imbalance.
  * precision@k — the practical question is "of the top k names flagged each
    day, how many actually pumped?", matching how a screener is used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .dataset import feature_columns

log = logging.getLogger(__name__)


def time_split(data: pd.DataFrame, cfg):
    dates = np.sort(data["date"].unique())
    n_test = max(int(len(dates) * float(cfg.model.test_fraction)), 1)
    purge = int(cfg.labeling.forward_window)
    test_dates = dates[-n_test:]
    train_dates = dates[: -(n_test + purge)]
    return data[data["date"].isin(train_dates)], data[data["date"].isin(test_dates)]


def precision_at_k(test: pd.DataFrame, scores: np.ndarray, k: int) -> float:
    """Mean daily precision among the top-k scored tickers."""
    t = test[["date", "label"]].copy()
    t["score"] = scores
    daily = []
    for _, g in t.groupby("date"):
        top = g.nlargest(min(k, len(g)), "score")
        daily.append(top["label"].mean())
    return float(np.mean(daily))


def train_model(data: pd.DataFrame, cfg) -> dict:
    feats = feature_columns(data)
    train, test = time_split(data, cfg)
    X_tr, y_tr = train[feats], train["label"]
    X_te, y_te = test[feats], test["label"]
    pos_rate = y_tr.mean()
    log.info("Train %d rows (%.2f%% pos) | Test %d rows (%.2f%% pos) | %d features",
             len(train), 100 * pos_rate, len(test), 100 * y_te.mean(), len(feats))

    # --- Baseline: logistic regression (median-imputed, scaled) ---
    med = X_tr.median(numeric_only=True)
    scaler = StandardScaler()
    Xtr_lr = scaler.fit_transform(X_tr.fillna(med))
    Xte_lr = scaler.transform(X_te.fillna(med))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(Xtr_lr, y_tr)
    lr_scores = lr.predict_proba(Xte_lr)[:, 1]

    # --- Main model: LightGBM (handles NaNs natively) ---
    # Early stopping on the most recent 15% of training dates (time-ordered,
    # purged) — without it the GBM overfits and loses to the linear baseline.
    p = cfg.model.lightgbm
    tr_dates = np.sort(train["date"].unique())
    n_val = max(int(len(tr_dates) * 0.15), 1)
    purge = int(cfg.labeling.forward_window)
    val_dates = tr_dates[-n_val:]
    fit_dates = tr_dates[: -(n_val + purge)]
    fit_mask = train["date"].isin(fit_dates).to_numpy()
    val_mask = train["date"].isin(val_dates).to_numpy()

    lgbm = LGBMClassifier(
        n_estimators=int(p.n_estimators), learning_rate=float(p.learning_rate),
        num_leaves=int(p.num_leaves), min_child_samples=int(p.min_child_samples),
        subsample=float(p.subsample), colsample_bytree=float(p.colsample_bytree),
        reg_lambda=float(p.reg_lambda),
        # No scale_pos_weight: we evaluate by ranking (PR-AUC/precision@k),
        # and reweighting hurt out-of-time ranking in the config sweep.
        random_state=42, verbose=-1,
    )
    from lightgbm import early_stopping, log_evaluation
    lgbm.fit(
        X_tr[fit_mask], y_tr[fit_mask],
        eval_set=[(X_tr[val_mask], y_tr[val_mask])], eval_metric="average_precision",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    # Refit at the chosen tree count on the full training window so the final
    # model sees the freshest data.
    best_n = lgbm.best_iteration_ or int(p.n_estimators)
    lgbm.set_params(n_estimators=best_n)
    lgbm.fit(X_tr, y_tr)
    metrics_extra = {"best_iteration": int(best_n)}
    lgbm_scores = lgbm.predict_proba(X_te)[:, 1]

    k = int(cfg.model.precision_at_k)
    base_rate = float(y_te.mean())
    metrics = {}
    for name, s in (("logistic", lr_scores), ("lightgbm", lgbm_scores)):
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_te, s)),
            "pr_auc": float(average_precision_score(y_te, s)),
            f"precision_at_{k}": precision_at_k(test, s, k),
        }
    metrics["test_base_rate"] = base_rate
    metrics.update(metrics_extra)

    imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    metrics["top_features"] = imp.head(15).round(4).to_dict()

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lgbm, "features": feats}, model_dir / "lgbm.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Saved model + metrics -> %s", model_dir)
    return metrics


def load_model(cfg):
    bundle = joblib.load(Path(cfg.paths.model_dir) / "lgbm.joblib")
    return bundle["model"], bundle["features"]
