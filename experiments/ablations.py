"""Both ablations the paper reports.

Architecture-level (paper Table 5): CNN-only, CNN+BiGRU, CNN+Transformer,
full CNN+Transformer+BiGRU, at identical preprocessing.

Preprocessing-level (paper Table 6): none, +MI only, +ADASYN only, +tau only,
MI+ADASYN, full. The toggles refit the pipeline per condition; threshold-off
conditions evaluate at 0.5.

The preprocessing ablation needs the real dataset (the toggles act on the
pipeline), so --smoke runs only the architecture arm on synthetic data.

Usage:
    python -m experiments.ablations data/msu
    python -m experiments.ablations --smoke
"""

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

import model as M
from experiments.baselines import _head, _train_keras, build_cnn_transformer
from experiments.common import prepare_variant, smoke_data, save_result, count_params
from experiments.run_crossval import fit_once
from train import tune_threshold, evaluate, BATCH


def build_cnn_only(t, f):
    inp = layers.Input(shape=(t, f))
    x = M._conv_block(inp, M.CONV_FILTERS[0])
    x = M._conv_block(x, M.CONV_FILTERS[1])
    return Model(inp, _head(x), name="cnn_only")


def build_cnn_bigru(t, f):
    inp = layers.Input(shape=(t, f))
    x = M._conv_block(inp, M.CONV_FILTERS[0])
    x = M._conv_block(x, M.CONV_FILTERS[1])
    x = layers.Bidirectional(layers.GRU(M.BIGRU_UNITS, return_sequences=True))(x)
    return Model(inp, _head(x), name="cnn_bigru")


ARCH_VARIANTS = {
    "cnn_only": build_cnn_only,
    "cnn_bigru": build_cnn_bigru,
    "cnn_transformer": build_cnn_transformer,
    "full": M.build_model,
}


def architecture_arm(data, epochs, batch):
    out = {}
    for name, builder in ARCH_VARIANTS.items():
        if name == "full":
            X_tr, y_tr, X_val, y_val, X_te, y_te = data
            m = fit_once(X_tr, y_tr, X_val, y_val, epochs, batch)
            t = tune_threshold(m, X_val, y_val)
            out[name] = {**evaluate(m, X_te, y_te, t),
                         "params": count_params(m)}
        else:
            metrics, _ = _train_keras(builder, data, epochs, batch)
            out[name] = {**metrics,
                         "params": count_params(builder(data[0].shape[1],
                                                        data[0].shape[2]))}
        print(f"{name}: f1={out[name]['f1']:.3f} "
              f"params={out[name]['params']['total']:,}")
    return out


def preprocessing_arm(csv_dir, epochs, batch):
    conditions = {
        "none":        dict(use_mi=False, use_adasyn=False, tune=False),
        "mi_only":     dict(use_mi=True,  use_adasyn=False, tune=False),
        "adasyn_only": dict(use_mi=False, use_adasyn=True,  tune=False),
        "tau_only":    dict(use_mi=False, use_adasyn=False, tune=True),
        "mi_adasyn":   dict(use_mi=True,  use_adasyn=True,  tune=False),
        "full":        dict(use_mi=True,  use_adasyn=True,  tune=True),
    }
    out = {}
    for name, cfg in conditions.items():
        data = prepare_variant(csv_dir, use_mi=cfg["use_mi"],
                               use_adasyn=cfg["use_adasyn"])
        X_tr, y_tr, X_val, y_val, X_te, y_te = data
        m = fit_once(X_tr, y_tr, X_val, y_val, epochs, batch)
        t = tune_threshold(m, X_val, y_val) if cfg["tune"] else 0.5
        out[name] = evaluate(m, X_te, y_te, t)
        print(f"{name}: f1={out[name]['f1']:.3f}")
    return out


def main(argv):
    smoke = "--smoke" in argv
    epochs, batch = (3, 64) if smoke else (200, BATCH)
    data = smoke_data() if smoke else prepare_variant(argv[0])

    payload = {"architecture": architecture_arm(data, epochs, batch),
               "paper_reports": "Tables 5 and 6 (not this run)"}
    if not smoke:
        payload["preprocessing"] = preprocessing_arm(argv[0], epochs, batch)
    save_result("ablations_smoke" if smoke else "ablations", payload)


if __name__ == "__main__":
    main(sys.argv[1:])
