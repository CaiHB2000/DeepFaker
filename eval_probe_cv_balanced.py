import numpy as np, pandas as pd, sys
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
assert "label" in df.columns
y = df["label"].astype(int).values
X = df.drop(columns=["path","label"]).values.astype(np.float32)

# 去 NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# 类别检查 & 均衡采样
vals, cnts = np.unique(y, return_counts=True)
print("[INFO] class counts:", dict(zip(vals, cnts)))
if len(vals) < 2:
    print("[FATAL] only one class in CSV. Please re-extract features with both classes.")
    sys.exit(1)
# 均衡到 min_count
m = cnts.min()
idx0 = np.where(y==0)[0][:m]
idx1 = np.where(y==1)[0][:m]
idx = np.concatenate([idx0, idx1])
X, y = X[idx], y[idx]
print("[INFO] balanced to:", {0: m, 1: m})

base_steps = [("var", VarianceThreshold(1e-8)), ("sc", StandardScaler())]

def cv_eval(clf, name):
    pipe = Pipeline(base_steps + [(name, clf)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=418)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X[te])[:,1]
        else:
            if hasattr(pipe, "decision_function"):
                d = pipe.decision_function(X[te])
                mn, mx = d.min(), d.max(); prob = (d - mn) / (mx - mn + 1e-9)
            else:
                prob = p.astype(float)
        aucs.append(roc_auc_score(y[te], prob))
    print(f"[CV] {name:18s} ACC={np.mean(accs):.4f}±{np.std(accs):.3f}  AUC={np.mean(aucs):.4f}±{np.std(aucs):.3f}")

# 1) LogReg（自动调 C）
cv_eval(LogisticRegressionCV(max_iter=5000, cv=5, n_jobs=-1, scoring="roc_auc", Cs=10, solver="lbfgs"), "LogRegCV")

# 2) RBF-SVM（小网格）
param_grid = {"C":[0.5,1,2,4,8], "gamma":["scale", 0.01, 0.005, 0.001]}
svc = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid=param_grid, cv=3, n_jobs=-1, scoring="roc_auc")
cv_eval(svc, "RBF-SVM(grid)")

# 3) LinearSVC + Platt 校准
cv_eval(CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000), cv=3), "Calibrated-LinearSVC")

# 4) RandomForest
cv_eval(RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=418), "RandomForest")

# Holdout30%
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=418)
best = Pipeline(base_steps + [("rbf", SVC(kernel="rbf", C=2, gamma="scale", probability=True))])
best.fit(Xtr, ytr)
pred = best.predict(Xte)
acc = accuracy_score(yte, pred)
auc = roc_auc_score(yte, best.predict_proba(Xte)[:,1])
print(f"[Holdout30%] RBF-SVM ACC={acc:.4f}  AUC={auc:.4f}")
