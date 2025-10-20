import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import sys
csv_path = sys.argv[1] if len(sys.argv)>1 else "/media/coolboy-3/chb/DeepFaker/datasets/np_features_vqdm_strong.csv"
df = pd.read_csv(csv_path)
y = df["label"].values.astype(int)
X = df.drop(columns=["path","label"]).values.astype(np.float32)

# 去 NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# 共同前处理：去常数列 + 标准化
base_steps = [
    ("var", VarianceThreshold(threshold=1e-8)),
    ("sc", StandardScaler())
]

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
            # decision_function -> sigmoid 近似
            if hasattr(pipe, "decision_function"):
                d = pipe.decision_function(X[te])
                # 统一到0-1（粗略）
                mn, mx = d.min(), d.max(); prob = (d - mn) / (mx - mn + 1e-9)
            else:
                prob = p.astype(float)
        aucs.append(roc_auc_score(y[te], prob))
    print(f"[CV] {name:18s} ACC={np.mean(accs):.4f}±{np.std(accs):.3f}  AUC={np.mean(aucs):.4f}±{np.std(aucs):.3f}")

# 1) LogReg（自动调C）
cv_eval(LogisticRegressionCV(max_iter=5000, cv=5, n_jobs=-1, scoring="roc_auc", Cs=10, solver="lbfgs"), "LogRegCV")

# 2) RBF-SVM（小网格）
param_grid = {"C":[0.5,1,2,4,8], "gamma":["scale", 0.01, 0.005, 0.001]}
svc = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid=param_grid, cv=3, n_jobs=-1, scoring="roc_auc")
cv_eval(svc, "RBF-SVM(grid)")

# 3) LinearSVC + Platt 校准
cv_eval(CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000), cv=3), "Calibrated-LinearSVC")

# 4) RandomForest（稳健 baseline）
cv_eval(RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=418), "RandomForest")

# 给一个 holdout 结果（便于和 D3 子集对齐）
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=418)
best = Pipeline(base_steps + [("rbf", SVC(kernel="rbf", C=2, gamma="scale", probability=True))])
best.fit(Xtr, ytr)
pred = best.predict(Xte)
acc = accuracy_score(yte, pred)
auc = roc_auc_score(yte, best.predict_proba(Xte)[:,1])
print(f"[Holdout30%] RBF-SVM ACC={acc:.4f}  AUC={auc:.4f}")
