"""Titanic survival classification analysis by Justin Spratt.

Uses an interpretable Logistic Regression pipeline and produces a confusion
matrix, ROC curve, ROC-AUC score, classification report, and top coefficients.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_clustering import load_titanic, NUM, CAT  # shared loader + columns


def run(data_path="CS379T-Week-1-IP.xls", out_dir="outputs"):
    """Returns metrics dict (ROC-AUC, classification report, top coefficients)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_titanic(data_path)
    x, y = df[NUM + CAT].copy(), df["survived"].astype(int)

    pre = ColumnTransformer(
        [
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
            ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                             ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT),
        ]
    )
    model = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, solver="liblinear"))])

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(x_tr, y_tr)

    pred = model.predict(x_te)
    proba = model.predict_proba(x_te)[:, 1]
    report = classification_report(y_te, pred, digits=3)
    auc = float(roc_auc_score(y_te, proba))

    cm = confusion_matrix(y_te, pred)
    plt.figure(); plt.imshow(cm)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")

    plt.figure(); RocCurveDisplay.from_predictions(y_te, proba)
    plt.savefig(out_dir / "roc_curve.png", dpi=160, bbox_inches="tight")

    feats = model.named_steps["pre"].get_feature_names_out()
    coefs = model.named_steps["lr"].coef_[0]
    top = sorted(zip(feats, coefs), key=lambda t: abs(t[1]), reverse=True)[:10]
    return {"roc_auc": auc, "classification_report": report, "top_coefficients": top}
