"""Encoder-only Transformer with dual output heads (direction + return).

Ports the architecture from the original research notebook
(``research/Deep Learning for Stock Price Prediction with LSTM and Transformers.py``,
lines 1141-1182) with two changes:

1. The notebook's ``y_output`` (Dense(1) regressor on next-day Close) and
   ``x_output`` (Dense(5) auxiliary feature reconstruction) are replaced with
   a *direction* head (``Dense(1, sigmoid)``) and a *return* head
   (``Dense(1, linear)``). This matches the production target schema:
   probability of up + expected return %.
2. Architecture dimensions are now driven entirely by ``hparams`` and the
   ``(seq_len, n_features)`` shape, instead of the notebook's hard-coded
   ``(60, 5)``. Same defaults — ``num_blocks=3, num_heads=4, key_dim=32,
   ffn_units=64, dense_units=32, dropout=0.1`` — read from
   ``src.config.TickerConfig.transformer_hparams``.

**v2/v3 attempt (2026-05-10) was reverted.** A pre-LN + residual rewrite with
input projection + AdamW degraded both regression (return_mae 4-7× worse)
and direction (delta_pp went from 0 to -1.98 on MSFT). Restored the verbatim
notebook block; the meaningful improvement that *did* survive is the
threshold-calibration step now baked into ``src/train.py`` (Youden-J on val
extracts the discriminative signal in the predicted probabilities — the
0.5-threshold reading was hiding it).
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam


def _transformer_block(
    x: tf.Tensor,
    *,
    num_heads: int,
    key_dim: int,
    ffn_units: int,
    dropout: float,
) -> tf.Tensor:
    """One encoder block: self-attention + feed-forward, each with residual-free
    Dropout → LayerNorm (matches the notebook's structure verbatim)."""
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = layers.Dropout(dropout)(attn)
    attn = layers.LayerNormalization()(attn)

    ffn = layers.Dense(ffn_units, activation="relu")(attn)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.LayerNormalization()(ffn)
    return ffn


def build_transformer(
    seq_len: int, n_features: int, hparams: dict[str, Any]
) -> Model:
    """Build and compile the dual-output Transformer.

    Output names are ``"direction"`` (sigmoid, BCE loss, accuracy + AUC) and
    ``"return"`` (linear, MSE loss, MAE metric). Train against a dict::

        model.fit(X, {"direction": y_dir, "return": y_ret}, ...)
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="sequence_input")

    # Positional encoding: learned Embedding over [0, seq_len), output dim
    # matched to n_features so it can be added to the input. `tf.range` is
    # rebuilt inside the graph so the model can be saved/restored cleanly.
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embed = layers.Embedding(
        input_dim=seq_len, output_dim=n_features, name="position_embedding"
    )(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    x = layers.Add(name="add_position")([inputs, pos_embed])

    for i in range(int(hparams["num_blocks"])):
        x = _transformer_block(
            x,
            num_heads=int(hparams["num_heads"]),
            key_dim=int(hparams["key_dim"]),
            ffn_units=int(hparams["ffn_units"]),
            dropout=float(hparams["dropout"]),
        )

    pooled = layers.GlobalAveragePooling1D(name="seq_pool")(x)
    shared = layers.Dense(
        int(hparams["dense_units"]), activation="relu", name="shared_dense"
    )(pooled)

    direction = layers.Dense(1, activation="sigmoid", name="direction")(shared)
    ret = layers.Dense(1, activation="linear", name="return")(shared)

    model = Model(inputs=inputs, outputs={"direction": direction, "return": ret})
    model.compile(
        optimizer=Adam(learning_rate=float(hparams["lr"])),
        loss={"direction": "binary_crossentropy", "return": "mse"},
        loss_weights={"direction": 1.0, "return": 1.0},
        metrics={
            "direction": ["accuracy", tf.keras.metrics.AUC(name="auc")],
            "return": ["mae"],
        },
    )
    return model
