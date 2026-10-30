"""Reconstruction. I wrote this in October 2026 from the paper's Methods
section; the original code did not survive. Steps 1-7 of
doi:10.1038/s41598-026-70577-4: load, clean, split, mutual information
selection, ADASYN, z-score, reshape.

Order matters and is the paper's central leakage claim:
split FIRST, then fit MI / ADASYN / scaler on the training partition only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek

SEED = 42
N_SELECTED = 60          # paper: top k = min(60, 128) by mutual information
TEST_FRACTION = 0.2      # paper: |D_test| = 0.2 |D|  (14,415 samples)
VAL_FRACTION = 0.1       # ASSUMPTION: validation slice size is not stated
T_STEPS = 10             # paper: equation (6) states T = 10, justified by SCADA
                         # polling intervals; 60 features -> shape (10, 6)
POSITIVE_LABEL = "Attack"  # ASSUMPTION: encoding direction is not stated


def load_msu(csv_dir):
    """15 benchmark CSVs -> one frame: 128 features + 'marker'."""
    frames = [pd.read_csv(p) for p in sorted(Path(csv_dir).glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    return df


def clean(df, label_col="marker"):
    """inf -> NaN -> drop rows; binary-encode the marker."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = (df[label_col].astype(str).str.strip() == POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[label_col]).astype(np.float32)
    return X, y.to_numpy()


def split(X, y):
    """Stratified 80/20, then a stratified validation slice off the training
    partition for early stopping and threshold tuning (never the test set)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=VAL_FRACTION, stratify=y_tr, random_state=SEED)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def select_features(X_tr, y_tr, *rest):
    """Mutual information, fitted on the training partition only."""
    mi = mutual_info_classif(X_tr, y_tr, random_state=SEED)
    top = np.argsort(mi)[::-1][:N_SELECTED]
    cols = X_tr.columns[top]
    return (X_tr[cols], *[X[cols] for X in rest]), cols


def balance(X_tr, y_tr):
    """ADASYN on the training partition only; SMOTETomek is the paper's
    documented fallback (never triggered in the published runs)."""
    try:
        return ADASYN(random_state=SEED).fit_resample(X_tr, y_tr)
    except (ValueError, RuntimeError):
        return SMOTETomek(random_state=SEED).fit_resample(X_tr, y_tr)


def standardize(X_tr, *rest):
    """Z-score fitted on the post-ADASYN training partition (per the paper)."""
    scaler = StandardScaler().fit(X_tr)
    return (scaler.transform(X_tr), *[scaler.transform(X) for X in rest]), scaler


def to_sequences(X):
    """(N, 60) -> (N, T, 60/T). Paper equation (6): f = floor(d'/T)."""
    n, d = X.shape
    f = d // T_STEPS
    return X[:, : T_STEPS * f].reshape(n, T_STEPS, f)


def prepare(csv_dir):
    X, y = clean(load_msu(csv_dir))
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    (X_tr, X_val, X_te), cols = select_features(X_tr, y_tr, X_val, X_te)
    X_tr, y_tr = balance(X_tr, y_tr)
    (X_tr, X_val, X_te), scaler = standardize(X_tr, X_val, X_te)
    return (to_sequences(X_tr), y_tr, to_sequences(X_val), y_val,
            to_sequences(X_te), y_te, cols, scaler)
