"""G-Flix anomaly detection. Run:  python anomaly_analysis.py
Needs anomaly_detection.csv in the same folder (or pass a path as argument).
Install:  pip install pandas numpy scipy scikit-learn matplotlib seaborn
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                      # saves PNGs, works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

CSV = sys.argv[1] if len(sys.argv) > 1 else "anomaly_detection.csv"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["remote"] = (df["remote_access"] == "Yes").astype(int)
num = ["login_duration_min", "data_accessed_MB", "files_downloaded"]
print(df.shape, "\nmissing:\n", df.isna().sum().to_string())
print(df[num].describe().to_string())

# ---------- EDA plots ----------
sns.set_style("whitegrid")
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, c in enumerate(num):
    sns.histplot(df[c], bins=40, ax=ax[0, i]); ax[0, i].set_title(c)
    sns.boxplot(x=df[c], ax=ax[1, i])
plt.tight_layout(); plt.savefig(f"{OUT}/01_distributions.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
sns.scatterplot(data=df, x="data_accessed_MB", y="files_downloaded", hue="remote_access", ax=ax[0], alpha=.7)
sns.scatterplot(data=df, x="login_duration_min", y="data_accessed_MB", hue="remote_access", ax=ax[1], alpha=.7)
sns.countplot(x="hour", data=df, ax=ax[2], color="steelblue")
plt.tight_layout(); plt.savefig(f"{OUT}/02_relationships.png", dpi=110); plt.close()

# ---------- Method 1: statistical ----------
df["zflag"] = (np.abs(stats.zscore(df[num])) > 3).any(axis=1)

med = df[num].median(); mad = (df[num] - med).abs().median()
df["rzflag"] = (0.6745 * (df[num] - med) / mad).abs().max(axis=1) > 3.5

iqr_flag = pd.Series(False, index=df.index)
for c in num:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    iqr_flag |= (df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)
df["iqrflag"] = iqr_flag

# ---------- Method 2: unsupervised ML (scaled!) ----------
X = StandardScaler().fit_transform(df[num + ["remote", "hour"]])
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42).fit(X)
df["if_score"] = -iso.score_samples(X)
df["ifflag"] = iso.predict(X) == -1
df["dbflag"] = DBSCAN(eps=2.0, min_samples=5).fit_predict(X) == -1

# ---------- Compare ----------
flags = ["zflag", "rzflag", "iqrflag", "ifflag", "dbflag"]
df["n_methods"] = df[flags].sum(axis=1)
print("\nFlag counts:\n", df[flags].sum().to_string())
top = df.sort_values(["n_methods", "if_score"], ascending=False).head(5)
print("\nTOP 5 SUSPECTS:\n", top[["timestamp", "user_id"] + num + ["remote_access", "n_methods", "if_score"]].to_string())

# ---------- Result plots ----------
P = PCA(2).fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(P[:, 0], P[:, 1], c="lightgrey", s=14, label="normal")
plt.scatter(*P[df.ifflag.values].T, marker="x", c="tab:blue", label="Isolation Forest")
plt.scatter(*P[df.dbflag.values].T, marker="*", c="red", s=140, label="DBSCAN")
for i in top.index: plt.annotate(df.loc[i, "user_id"], P[i], fontsize=8)
plt.legend(); plt.title("PCA view of flagged points")
plt.savefig(f"{OUT}/04_pca_flags.png", dpi=110); plt.close()

M = np.array([[(df[a] & df[b]).sum() for b in flags] for a in flags])
plt.figure(figsize=(5, 4)); sns.heatmap(M, annot=True, fmt="d", xticklabels=flags, yticklabels=flags, cmap="Blues")
plt.tight_layout(); plt.savefig(f"{OUT}/05_overlap.png", dpi=110); plt.close()

df.sort_values("if_score", ascending=False).to_csv("anomaly_results_all_records.csv", index=False)
print("\nSaved figs/ and anomaly_results_all_records.csv")
