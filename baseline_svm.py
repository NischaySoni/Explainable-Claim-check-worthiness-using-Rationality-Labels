"""
Baseline 1: SVM + TF-IDF for Binary Claim Check-Worthiness Detection
=====================================================================
Uses TF-IDF features from the 'claim' column and trains a LinearSVC
classifier for binary check-worthiness (check-worthy vs non-check-worthy).
Also trains per-label LinearSVC classifiers for the 6 rationality labels
(evaluated only on check-worthy subset).
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# ─── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val_data.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_data.csv")

# ─── Rationality Columns ──────────────────────────────────────────────
RATIONALITY_COLS = [
    "verifiable_factual_claim",
    "false_info",
    "general_public_interest",
    "harmful",
    "fact_checker_interest",
    "govt_interest",
]

# ─── Load Data ────────────────────────────────────────────────────────
print("=" * 60)
print("Baseline 1: SVM + TF-IDF")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ─── Binary labels ────────────────────────────────────────────────────
X_train = train_df["claim"].fillna("").astype(str)
y_train = train_df["bin_label"].values

X_val   = val_df["claim"].fillna("").astype(str)
y_val   = val_df["bin_label"].values

X_test  = test_df["claim"].fillna("").astype(str)
y_test  = test_df["bin_label"].values

# ─── Rationality labels ───────────────────────────────────────────────
def get_rat(df):
    """Return rationality label array; zero out rows where bin_label == 0."""
    arr = df[RATIONALITY_COLS].fillna(0).values.astype(np.int64)
    arr[df["bin_label"].values == 0] = 0
    return arr

y_rat_train = get_rat(train_df)
y_rat_val   = get_rat(val_df)
y_rat_test  = get_rat(test_df)

# ─── TF-IDF Vectorization ─────────────────────────────────────────────
print("\nFitting TF-IDF vectorizer ...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

# ─── Train Binary SVM ─────────────────────────────────────────────────
print("Training binary LinearSVC ...")
svm = LinearSVC(
    class_weight="balanced",
    max_iter=10000,
    C=1.0,
    random_state=42,
)
svm.fit(X_train_tfidf, y_train)

# ─── Train Per-Label Rationality SVMs ─────────────────────────────────
# Only trained on check-worthy samples (bin_label == 1)
print("Training per-label rationality SVMs (on check-worthy training subset) ...")
cw_mask_train = y_train == 1
X_train_cw    = X_train_tfidf[cw_mask_train]

rat_svms = []
for i, col in enumerate(RATIONALITY_COLS):
    clf = LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train_cw, y_rat_train[cw_mask_train, i])
    rat_svms.append(clf)
    print(f"  Trained rationality SVM for: {col}")

# ─── Evaluate ─────────────────────────────────────────────────────────
def evaluate(name, X, y_true, y_rat_true):
    """Evaluate binary SVM and per-label rationality SVMs."""

    # ── Binary check-worthiness ──
    y_pred   = svm.predict(X)
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cw_f1    = f1_score(y_true, y_pred, pos_label=1)
    ncw_f1   = f1_score(y_true, y_pred, pos_label=0)

    print(f"\n{'─' * 50}")
    print(f"  {name} Results — Binary Check-Worthiness")
    print(f"{'─' * 50}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  CW F1 (cls=1)  : {cw_f1:.4f}")
    print(f"  NCW F1 (cls=0) : {ncw_f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(
        classification_report(
            y_true, y_pred,
            target_names=["Non-Check-Worthy", "Check-Worthy"],
        )
    )

    # ── Rationality labels (check-worthy subset only) ──
    cw_mask  = y_true == 1
    X_cw     = X[cw_mask]
    rat_true = y_rat_true[cw_mask]

    print(f"  {name} Results — Rationality Labels (check-worthy subset, n={cw_mask.sum()})")
    print(f"  {'Label':<35s}  {'Macro F1':>9}")
    print(f"  {'─'*46}")

    rat_f1s = []
    for i, col in enumerate(RATIONALITY_COLS):
        rat_pred = rat_svms[i].predict(X_cw)
        f1 = f1_score(
            rat_true[:, i], rat_pred,
            average="macro", zero_division=0,
        )
        rat_f1s.append(f1)
        print(f"  {col:<35s}: {f1:.4f}")

    return acc, macro_f1, cw_f1, ncw_f1, rat_f1s


val_metrics  = evaluate("Validation", X_val_tfidf,  y_val,  y_rat_val)
test_metrics = evaluate("Test",        X_test_tfidf, y_test, y_rat_test)

# ─── Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("  Summary: SVM + TF-IDF Baseline")
print("=" * 95)
header = (
    f"  {'Split':<12} {'Acc':>6} {'m-F1':>6} {'cw-F1':>7} {'ncw-F1':>8}  "
    + "  ".join(f"R{i+1:>1}" for i in range(6))
)
print(header)
print(f"  {'─' * 88}")
for split_name, m in [("Validation", val_metrics), ("Test", test_metrics)]:
    r = m[4]
    rat_str = "  ".join(f"{v:.4f}" for v in r)
    print(
        f"  {split_name:<12} {m[0]:>6.4f} {m[1]:>6.4f} {m[2]:>7.4f} {m[3]:>8.4f}  {rat_str}"
    )
print("=" * 95)
