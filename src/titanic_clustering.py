"""Titanic passenger clustering analysis by Justin Spratt.

Uses K-Means clustering, silhouette scoring, and PCA visualization to
describe passenger groups. Outputs include silhouette scores, a two-dimensional
cluster plot, and a cluster profile table.
"""
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["pclass", "age", "sibsp", "parch", "fare"]
CAT = ["sex", "embarked"]


def load_titanic(path):
    """Reads .xls/.xlsx; if .xls needs xlrd, tries converting to .xlsx via LibreOffice."""
    path = Path(path)
    try:
        return pd.read_excel(path)
    except ImportError as exc:
        if path.suffix.lower() != ".xls":
            raise
        xlsx = path.with_suffix(".xlsx")
        cmd = ["soffice", "--headless", "--convert-to", "xlsx", "--outdir", str(path.parent), str(path)]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if xlsx.exists():
            return pd.read_excel(xlsx)
        raise ImportError("Install xlrd or provide a .xlsx dataset copy.") from exc


def run(data_path="CS379T-Week-1-IP.xls", out_dir="outputs"):
    """Returns (best_k, silhouette_by_k, cluster_profile_df)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_titanic(data_path)

    pre = ColumnTransformer(
        [
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
            ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                             ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT),
        ]
    )
    xt = pre.fit_transform(df[NUM + CAT])
    xt = xt.toarray() if hasattr(xt, "toarray") else xt

    ks = list(range(2, 7))
    sil = [silhouette_score(xt, KMeans(k, random_state=42, n_init=10).fit_predict(xt)) for k in ks]
    sil_by_k = dict(zip(ks, map(float, sil)))
    best_k = max(sil_by_k, key=sil_by_k.get)

    plt.figure(); plt.plot(ks, sil, marker="o")
    plt.savefig(out_dir / "silhouette_scores.png", dpi=160, bbox_inches="tight")

    labels = KMeans(best_k, random_state=42, n_init=10).fit_predict(xt)
    prof = df[NUM + CAT].copy()
    prof["cluster"] = labels
    prof["sex"] = prof["sex"].fillna("Unknown")
    prof["embarked"] = prof["embarked"].fillna("Unknown")
    profile = pd.concat(
        [
            prof["cluster"].value_counts().sort_index().rename("n"),
            prof.groupby("cluster")[NUM].mean(),
            pd.crosstab(prof["cluster"], prof["sex"], normalize="index").add_prefix("sex_"),
            pd.crosstab(prof["cluster"], prof["embarked"], normalize="index").add_prefix("emb_"),
        ],
        axis=1,
    ).round(3)
    profile.to_csv(out_dir / "cluster_profile.csv")

    p2 = PCA(n_components=2, random_state=42).fit_transform(xt)
    plt.figure(); plt.scatter(p2[:, 0], p2[:, 1], c=labels)
    plt.savefig(out_dir / "clusters_pca.png", dpi=160, bbox_inches="tight")
    return best_k, sil_by_k, profile
