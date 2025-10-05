import pandas as pd, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from src.features import to_features, RISK_FEATURES
from src.model import synthesize_labels
from src.pricing import risk_score, premium_from_score

# import numpy as np
# rng = np.random.default_rng(42)  # same seed
# np.random.seed(42)               # for any legacy calls
import numpy as np
np.random.seed(42)



# load events and build features
df = pd.read_json("data/events.ndjson", lines=True)
feats = to_features(df)

# labels for POC evaluation
# replace usages of np.random.* with rng.* if needed
from src.model import synthesize_labels
y = synthesize_labels(feats)  # already seeded now

#y = synthesize_labels(feats)
X = feats[RISK_FEATURES].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# GLM baseline
glm = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
p_glm = glm.predict_proba(Xte)[:,1]

# GBM (+ adaptive calibration)
gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08)
min_class = min(int(ytr.sum()), int(len(ytr)-ytr.sum()))
if min_class >= 3:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = CalibratedClassifierCV(gbm, method="isotonic", cv=cv).fit(Xtr, ytr)
elif min_class >= 2:
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    model = CalibratedClassifierCV(gbm, method="sigmoid", cv=cv).fit(Xtr, ytr)
else:
    model = gbm.fit(Xtr, ytr)
p_gbm = model.predict_proba(Xte)[:,1]

def metrics(name, ytrue, p):
    return {
        "AUC": float(roc_auc_score(ytrue,p)),
        "Brier": float(brier_score_loss(ytrue,p)),
        "LogLoss": float(log_loss(ytrue,p))
    }

m_glm  = metrics("GLM", yte, p_glm)
m_gbm  = metrics("GBM/Cal", yte, p_gbm)
print("GLM    :", m_glm)
print("GBM/Cal:", m_gbm)

# Calibration plot
prob_true, prob_pred = calibration_curve(yte, p_gbm, n_bins=10, strategy="quantile")
plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0,1],[0,1],"--")
plt.title("Calibration: GBM/Cal")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/calibration.png", bbox_inches="tight")

# Simple ROI demo
base = 120.0
scores = np.array([risk_score(p) for p in p_gbm])
prem_dyn = np.array([premium_from_score(base, s) for s in scores])
EL = p_gbm * 100  # demo loss scale
import pandas as pd
pd.DataFrame({
    "risk_score": scores, "prob": p_gbm,
    "premium_dynamic": prem_dyn, "expected_loss": EL
}).to_csv("docs/roi_summary.csv", index=False)

print("Saved docs/calibration.png and docs/roi_summary.csv")
