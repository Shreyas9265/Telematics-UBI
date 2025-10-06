import argparse, pandas as pd, numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
import sys, os
sys.path.append(os.path.dirname(__file__))  # add current folder to path
from features import to_features, RISK_FEATURES


def synthesize_labels(df):
    import numpy as np
    rng = np.random.default_rng(42)   # fixed seed
    w = np.array([1.6, 1.2, 1.1, 0.9, 1.0, 0.6])
    z = df[RISK_FEATURES].values @ w - 2.0 + rng.normal(0, 0.25, len(df))
    p = 1/(1+np.exp(-z))
    return (rng.random(len(df)) < p).astype(int)




def _calibrator_or_gbm(gbm, X, y):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    min_class = min(n_pos, n_neg)

    # Fit GBM first; we may return it if calibration isn't feasible
    gbm.fit(X, y)

    if min_class >= 3:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cal = CalibratedClassifierCV(gbm, method="isotonic", cv=cv)
        cal.fit(X, y)
        return cal
    elif min_class >= 2:
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        cal = CalibratedClassifierCV(gbm, method="sigmoid", cv=cv)
        cal.fit(X, y)
        return cal
    else:
        # Too few positives/negatives; skip calibration
        return gbm

def train(feats: pd.DataFrame):
    y = synthesize_labels(feats)
    X = feats[RISK_FEATURES].values

    glm = LogisticRegression(max_iter=1000)
    glm.fit(X, y)

    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08)
    model = _calibrator_or_gbm(gbm, X, y)

    def score(m):
        proba = m.predict_proba(X)[:,1] if hasattr(m,"predict_proba") else m.predict(X)
        return {
            "AUC": float(roc_auc_score(y, proba)),
            "Brier": float(brier_score_loss(y, proba)),
            "LogLoss": float(log_loss(y, proba))
        }

    metrics = {"GLM": score(glm), "GBM_or_Cal": score(model)}
    joblib.dump(glm, "models/glm.joblib")
    joblib.dump(model, "models/model.joblib")
    return metrics

def load_model():
    return joblib.load("models/model.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to events NDJSON")
    args = ap.parse_args()
    df = pd.read_json(args.train, lines=True)
    feats = to_features(df)
    m = train(feats)
    print("Metrics:", m)
