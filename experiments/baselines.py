"""The five baselines the paper retrains inside the same pipeline, plus
McNemar's test against the full model.

Paper Table 4 baselines: Random Forest, LSTM, CNN-LSTM, Transformer,
CNN-Transformer. The paper does not state the baselines' widths either, so
these use the same assumption scale as model.py (LSTM 128 units, conv blocks
64/128). Treat every number this produces as a new result.

Usage:
    python -m experiments.baselines data/msu
    python -m experiments.baselines --smoke
"""

import sys

import numpy as np
import tensorflow as tf
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, Model

import model as M
from experiments.common import SEED, prepare_variant, smoke_data, save_result
from experiments.run_crossval import fit_once
from train import tune_threshold, evaluate, BATCH


def _head(x):
    pooled = layers.Concatenate()([layers.GlobalAveragePooling1D()(x),
                                   layers.GlobalMaxPooling1D()(x)])
    h = layers.Dense(M.HEAD_UNITS, activation="swish",
                     kernel_regularizer=tf.keras.regularizers.l2(M.HEAD_L2))(pooled)
    h = layers.Dropout(M.DROPOUT)(h)
    return layers.Dense(1, activation="sigmoid", dtype="float32")(h)


def build_lstm(t, f):
    inp = layers.Input(shape=(t, f))
    x = layers.LSTM(128, return_sequences=True)(inp)
    return Model(inp, _head(x), name="lstm_baseline")


def build_cnn_lstm(t, f):
    inp = layers.Input(shape=(t, f))
    x = M._conv_block(inp, M.CONV_FILTERS[0])
    x = M._conv_block(x, M.CONV_FILTERS[1])
    x = layers.LSTM(128, return_sequences=True)(x)
    return Model(inp, _head(x), name="cnn_lstm_baseline")


def build_transformer(t, f):
    inp = layers.Input(shape=(t, f))
    x = layers.Dense(M.CONV_FILTERS[1])(inp)   # project up so MHA has width
    x = M._transformer_block(x)
    return Model(inp, _head(x), name="transformer_baseline")


def build_cnn_transformer(t, f):
    inp = layers.Input(shape=(t, f))
    x = M._conv_block(inp, M.CONV_FILTERS[0])
    x = M._conv_block(x, M.CONV_FILTERS[1])
    x = M._transformer_block(x)
    return Model(inp, _head(x), name="cnn_transformer_baseline")


BUILDERS = {
    "lstm": build_lstm,
    "cnn_lstm": build_cnn_lstm,
    "transformer": build_transformer,
    "cnn_transformer": build_cnn_transformer,
}


def mcnemar(y_true, pred_a, pred_b):
    """Continuity-corrected McNemar on the discordant pairs, paper eq. 15."""
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b_ = int(np.sum(~a_right & b_right))
    c_ = int(np.sum(a_right & ~b_right))
    if b_ + c_ == 0:
        return {"b": b_, "c": c_, "chi2": 0.0, "p": 1.0}
    stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
    return {"b": b_, "c": c_, "chi2": float(stat),
            "p": float(chi2.sf(stat, df=1))}


def _train_keras(builder, data, epochs, batch):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    m = builder(X_tr.shape[1], X_tr.shape[2])
    steps = int(np.ceil(len(X_tr) / batch))
    sched = tf.keras.optimizers.schedules.CosineDecay(
        1e-3, decay_steps=max(1, steps * epochs), alpha=1e-3)
    m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=sched,
                                                  weight_decay=1e-4),
              loss="binary_crossentropy")
    m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
          batch_size=batch, verbose=0)
    t = tune_threshold(m, X_val, y_val)
    p = m.predict(X_te, batch_size=batch, verbose=0).ravel()
    return evaluate(m, X_te, y_te, t), (p >= t).astype(int)


def main(argv):
    smoke = "--smoke" in argv
    epochs, batch = (3, 64) if smoke else (200, BATCH)
    data = smoke_data() if smoke else prepare_variant(argv[0])
    X_tr, y_tr, X_val, y_val, X_te, y_te = data

    results, preds = {}, {}

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    rf_pred = rf.predict(X_te.reshape(len(X_te), -1))
    rf_prob = rf.predict_proba(X_te.reshape(len(X_te), -1))[:, 1]
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 roc_auc_score)
    results["random_forest"] = {
        "accuracy": accuracy_score(y_te, rf_pred),
        "f1": f1_score(y_te, rf_pred),
        "mcc": matthews_corrcoef(y_te, rf_pred),
        "roc_auc": roc_auc_score(y_te, rf_prob)}
    preds["random_forest"] = rf_pred

    for name, builder in BUILDERS.items():
        results[name], preds[name] = _train_keras(builder, data, epochs, batch)
        print(f"{name}: f1={results[name]['f1']:.3f}")

    ours = fit_once(X_tr, y_tr, X_val, y_val, epochs, batch)
    t = tune_threshold(ours, X_val, y_val)
    results["ours_full"] = evaluate(ours, X_te, y_te, t)
    ours_pred = (ours.predict(X_te, batch_size=batch, verbose=0).ravel()
                 >= t).astype(int)

    tests = {name: mcnemar(y_te, ours_pred, p) for name, p in preds.items()}
    save_result("baselines_smoke" if smoke else "baselines",
                {"metrics": results, "mcnemar_vs_ours": tests,
                 "paper_reports": "Table 4 (not this run)"})


if __name__ == "__main__":
    main(sys.argv[1:])
