"""Shared pieces for the experiment scripts.

Everything under experiments/ is NEW code, written after publication against
the reconstruction. Nothing it produces is a published number. Results land
in experiments/results/ as JSON, each stamped with that warning.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from preprocess import (SEED, VAL_FRACTION, T_STEPS, clean, load_msu, split,
                        select_features, balance, standardize, to_sequences)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

NOTE = ("New result from the October 2026 reconstruction. Not a published "
        "number and not produced by the original experiment code, which did "
        "not survive.")


def prepare_variant(csv_dir, use_mi=True, use_adasyn=True):
    """The pipeline with MI and ADASYN individually switchable, for the
    preprocessing ablation. use_mi=False keeps all 128 features (reshaped
    16 x 8 so T stays close to the paper's 10)."""
    X, y = clean(load_msu(csv_dir))
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    if use_mi:
        (X_tr, X_val, X_te), cols = select_features(X_tr, y_tr, X_val, X_te)
    if use_adasyn:
        X_tr, y_tr = balance(X_tr, y_tr)
    (X_tr, X_val, X_te), _ = standardize(X_tr, X_val, X_te)
    seq = to_sequences if use_mi else _to_sequences_128
    return seq(X_tr), y_tr, seq(X_val), y_val, seq(X_te), y_te


def _to_sequences_128(X):
    n, d = np.asarray(X).shape
    t = 16
    f = d // t
    return np.asarray(X)[:, : t * f].reshape(n, t, f)


def smoke_data(n=600, timesteps=T_STEPS, features=6, seed=SEED):
    """Tiny synthetic dataset with a planted linear signal, for checking that
    the experiment code runs end to end. Scores on it mean nothing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, timesteps, features)).astype("float32")
    w = rng.normal(size=(timesteps * features,))
    logits = X.reshape(n, -1) @ w * 0.5
    y = (logits + rng.normal(scale=1.0, size=n) > 0).astype(int)
    k = int(0.7 * n)
    v = int(0.85 * n)
    return X[:k], y[:k], X[k:v], y[k:v], X[v:], y[v:]


def save_result(name, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"note": NOTE, "written": time.strftime("%Y-%m-%d %H:%M"),
               **payload}
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return out


def count_params(model):
    tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    nt = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"trainable": tr, "non_trainable": nt, "total": tr + nt}
