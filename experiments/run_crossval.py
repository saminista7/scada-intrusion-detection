"""Five-fold stratified cross-validation of the full model.

The paper reports F1 = 0.87 +/- 0.01, MCC = 0.73 +/- 0.012, ROC-AUC =
0.93 +/- 0.008 over five stratified folds. This script runs that protocol
against the reconstruction: the entire pipeline (MI selection, ADASYN,
scaler, threshold) is refitted inside every fold, so nothing fitted on one
fold's test rows ever touches them.

Usage:
    python -m experiments.run_crossval data/msu
    python -m experiments.run_crossval --smoke     # synthetic end-to-end check
"""

import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.common import (SEED, clean, load_msu, save_result,
                                select_features, balance, standardize,
                                to_sequences, smoke_data, VAL_FRACTION)
from model import build_model
from train import ValF1EarlyStopping, class_weights, tune_threshold, evaluate, BATCH

import tensorflow as tf

N_FOLDS = 5


def fit_once(X_tr, y_tr, X_val, y_val, epochs, batch):
    model = build_model(X_tr.shape[1], X_tr.shape[2])
    steps = int(np.ceil(len(X_tr) / batch))
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        1e-3, decay_steps=max(1, steps * epochs), alpha=1e-3)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule,
                                                      weight_decay=1e-4),
                  loss="binary_crossentropy")
    es = ValF1EarlyStopping(X_val, y_val)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=batch, class_weight=class_weights(y_tr),
              callbacks=[es], verbose=0)
    return model


def main(argv):
    smoke = "--smoke" in argv
    epochs, batch = (3, 64) if smoke else (200, BATCH)

    if smoke:
        X_tr, y_tr, X_val, y_val, X_te, y_te = smoke_data()
        X = np.concatenate([X_tr, X_val, X_te])
        y = np.concatenate([y_tr, y_val, y_te])
        folds, scores = 2, []
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        for i, (tr, te) in enumerate(skf.split(X.reshape(len(X), -1), y)):
            Xi, Xt = X[tr], X[te]
            yi, yt = y[tr], y[te]
            Xi, Xv, yi, yv = train_test_split(Xi, yi, test_size=0.2,
                                              stratify=yi, random_state=SEED)
            m = fit_once(Xi, yi, Xv, yv, epochs, batch)
            t = tune_threshold(m, Xv, yv)
            scores.append(evaluate(m, Xt, yt, t))
            print(f"fold {i + 1}/{folds}: f1={scores[-1]['f1']:.3f}")
    else:
        csv_dir = argv[0]
        X, y = clean(load_msu(csv_dir))
        scores = []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            X_fold, y_fold = X.iloc[tr_idx], y[tr_idx]
            X_te, y_te = X.iloc[te_idx], y[te_idx]
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_fold, y_fold, test_size=VAL_FRACTION, stratify=y_fold,
                random_state=SEED)
            X_tr_s, y_tr_b, X_val_s, X_te_s = _fit_fold_pipeline(
                X_tr, y_tr, X_val, X_te)
            m = fit_once(X_tr_s, y_tr_b, X_val_s, y_val, epochs, batch)
            t = tune_threshold(m, X_val_s, y_val)
            scores.append(evaluate(m, X_te_s, y_te, t))
            print(f"fold {i + 1}/{N_FOLDS}: f1={scores[-1]['f1']:.3f}")

    agg = {k: {"mean": float(np.mean([s[k] for s in scores])),
               "std": float(np.std([s[k] for s in scores]))}
           for k in scores[0]}
    save_result("crossval_smoke" if smoke else "crossval",
                {"folds": scores, "aggregate": agg,
                 "paper_reports": "F1 0.87+/-0.01, MCC 0.73+/-0.012, "
                                  "ROC-AUC 0.93+/-0.008 (not this run)"})


def _fit_fold_pipeline(X_tr, y_tr, X_val, X_te):
    (X_tr, X_val, X_te), _ = select_features(X_tr, y_tr, X_val, X_te)
    X_tr, y_tr_b = balance(X_tr, y_tr)
    (X_tr, X_val, X_te), _ = standardize(X_tr, X_val, X_te)
    return to_sequences(X_tr), y_tr_b, to_sequences(X_val), to_sequences(X_te)


if __name__ == "__main__":
    main(sys.argv[1:])
