"""Reconstruction. I wrote this in October 2026 from the paper's Methods
section; the original code did not survive. Training protocol and evaluation
for doi:10.1038/s41598-026-70577-4.

Stated by the paper: weighted BCE with w_c = N / (C * N_c); AdamW, base LR
1e-3, weight decay 1e-4, cosine annealing to 1e-6 plus ReduceLROnPlateau
(factor 0.3, patience 6); batch 256; up to 200 epochs with early stopping on
validation F1 (patience 12); decision threshold tuned by grid search over
[0.1, 0.9] step 0.01 maximizing F1 on the validation slice, never the test set.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix)

from model import build_model
from preprocess import prepare

BATCH = 256
MAX_EPOCHS = 200
ES_PATIENCE = 12
BASE_LR = 1e-3
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4


class ValF1EarlyStopping(tf.keras.callbacks.Callback):
    """Early stopping on validation F1 at threshold 0.5 (tuning the threshold
    happens once, after training). Restores the best weights."""

    def __init__(self, X_val, y_val, patience=ES_PATIENCE):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.patience = patience
        self.best, self.wait, self.best_weights = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X_val, batch_size=BATCH, verbose=0).ravel()
        f1 = f1_score(self.y_val, (p >= 0.5).astype(int))
        if f1 > self.best:
            self.best, self.wait = f1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        (logs or {})["val_f1"] = f1

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def class_weights(y):
    """w_c = N / (C * N_c), paper equation (12)."""
    n = len(y)
    return {c: n / (2 * np.sum(y == c)) for c in (0, 1)}


def tune_threshold(model, X_val, y_val):
    """Grid over [0.1, 0.9] step 0.01 maximizing F1 on the VALIDATION slice."""
    p = model.predict(X_val, batch_size=BATCH, verbose=0).ravel()
    grid = np.arange(0.10, 0.901, 0.01)
    f1s = [f1_score(y_val, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(model, X_te, y_te, threshold):
    p = model.predict(X_te, batch_size=BATCH, verbose=0).ravel()
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_te, yhat),
        "precision": precision_score(y_te, yhat),
        "recall": recall_score(y_te, yhat),
        "f1": f1_score(y_te, yhat),
        "mcc": matthews_corrcoef(y_te, yhat),
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),
        "fnr": fn / (fn + tp),   # paper reports 9%
        "far": fp / (fp + tn),   # paper reports 7%
    }


def main(csv_dir, steps_per_epoch_hint=None):
    X_tr, y_tr, X_val, y_val, X_te, y_te, cols, _ = prepare(csv_dir)

    model = build_model(X_tr.shape[1], X_tr.shape[2])
    steps = steps_per_epoch_hint or int(np.ceil(len(X_tr) / BATCH))
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        BASE_LR, decay_steps=steps * MAX_EPOCHS, alpha=MIN_LR / BASE_LR)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule,
                                            weight_decay=WEIGHT_DECAY),
        loss="binary_crossentropy")

    # The paper uses both cosine annealing and ReduceLROnPlateau (factor 0.3,
    # patience 6). Keras cannot wrap a schedule in ReduceLROnPlateau, so
    # composing the two faithfully needs a custom callback. I note it here
    # instead of quietly approximating it.
    es = ValF1EarlyStopping(X_val, y_val)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=MAX_EPOCHS, batch_size=BATCH,
              class_weight=class_weights(y_tr), callbacks=[es], verbose=2)

    threshold = tune_threshold(model, X_val, y_val)
    metrics = evaluate(model, X_te, y_te, threshold)
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/msu")
