"""Reconstruction. I wrote this in October 2026 from the paper's Methods
section; the original code did not survive. The three-stage
CNN-Transformer-BiGRU of doi:10.1038/s41598-026-70577-4, section
"Hybrid deep learning architecture".

Stated by the paper: two Conv1D blocks with Swish and BatchNorm; one
Transformer encoder block with 8 heads and d_k = 64 (deeper stacks overfit);
BiGRU with 128 units per direction; [GlobalAvgPool; GlobalMaxPool] ->
Dense(256, L2 = 1e-4) -> sigmoid. Everything marked ASSUMPTION is not in
the paper.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

N_HEADS = 8          # paper
D_K = 64             # paper
BIGRU_UNITS = 128    # paper: per direction
HEAD_UNITS = 256     # paper
HEAD_L2 = 1e-4       # paper

# ASSUMPTIONS. These four are not in the paper, and they are what the
# parameter count is almost entirely made of. As set here the model has
# 686,273 trainable parameters (686,657 total once BatchNorm's running
# statistics are counted), about 46% of the ~1.5M the paper reports -- so
# this is demonstrably NOT the published network. Sweeping the three width
# assumptions over filters {32,64,128,256}, kernels {3,5,7} and FFN
# {128,256,512,1024}, every setting that lands within 5% of 1.5M has 256
# filters in the second conv block (totals: (128,256) kernel 7 FFN 512 ->
# 1,454,721; (32,256) kernel 3 FFN 1024 -> 1,507,297). The published model
# was wider here than this reconstruction is. Left as-is rather than tuned
# to hit the number, because fitting the assumptions to the headline figure
# would manufacture agreement instead of measuring it.
CONV_FILTERS = (64, 128)  # ASSUMPTION: filter counts not stated
KERNEL_SIZE = 3           # ASSUMPTION: kernel size not stated
FFN_UNITS = 256           # ASSUMPTION: transformer FFN width not stated
DROPOUT = 0.2             # ASSUMPTION: block dropout not stated


def _conv_block(x, filters):
    """H = Swish(BN(W * H + b)), paper equation (7)."""
    x = layers.Conv1D(filters, KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("swish")(x)


def _transformer_block(x):
    """Single pre-norm encoder block, 8 heads, d_k 64 (paper equations 8a-8b).
    ASSUMPTION: pre-norm vs post-norm is not stated; pre-norm used."""
    d_model = x.shape[-1]
    h = layers.LayerNormalization()(x)
    h = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=D_K,
                                  dropout=DROPOUT)(h, h)
    x = layers.Add()([x, h])
    h = layers.LayerNormalization()(x)
    h = layers.Dense(FFN_UNITS, activation="swish")(h)
    h = layers.Dropout(DROPOUT)(h)
    h = layers.Dense(d_model)(h)
    return layers.Add()([x, h])


def build_model(timesteps, features):
    inp = layers.Input(shape=(timesteps, features))

    # Stage 1: CNN local temporal motifs
    x = _conv_block(inp, CONV_FILTERS[0])
    x = _conv_block(x, CONV_FILTERS[1])

    # Stage 2: single Transformer encoder block, global context
    x = _transformer_block(x)

    # Stage 3: BiGRU, bidirectional sequence representation
    x = layers.Bidirectional(
        layers.GRU(BIGRU_UNITS, return_sequences=True))(x)

    # Head: dual pooling -> regularized Dense -> sigmoid (paper equation 10)
    pooled = layers.Concatenate()([
        layers.GlobalAveragePooling1D()(x),
        layers.GlobalMaxPooling1D()(x),
    ])
    h = layers.Dense(HEAD_UNITS, activation="swish",
                     kernel_regularizer=regularizers.l2(HEAD_L2))(pooled)
    h = layers.Dropout(DROPOUT)(h)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(h)
    return Model(inp, out, name="cnn_transformer_bigru_RECONSTRUCTION")


if __name__ == "__main__":
    m = build_model(timesteps=10, features=6)
    m.summary()
