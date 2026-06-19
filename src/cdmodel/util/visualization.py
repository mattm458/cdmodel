from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from cdmodel.model.types import AttentionMaskingStrategy


def plot_weights(
    weights: ArrayLike,
    masking_strategy: AttentionMaskingStrategy,
    att_style: Literal["single"] | Literal["dual"],
    spk_side: ArrayLike,
    spk: int | None,
    title: str,
    logger,
    global_step: int,
):
    matplotlib.use("Agg")

    if att_style == "single":
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.set_tight_layout(True)

        if spk is not None:
            timestep_mask = spk_side[1:] == spk
            if masking_strategy == "partner":
                history_mask = spk_side != spk
            elif masking_strategy == "self":
                history_mask = spk_side == spk
            elif masking_strategy == "both":
                history_mask = np.ones_like(spk_side, dtype=np.bool)
        else:
            timestep_mask = np.ones_like(spk_side[1:], dtype=np.bool)
            history_mask = np.ones_like(spk_side, dtype=np.bool)

        weights = weights[0, timestep_mask]
        weights = np.array([x[history_mask] for x in np.unstack(weights, axis=0)])
        weights = (weights - weights.min(0)) / (
            ((weights.max(0) - weights.min(0))) + 1e-10
        )

        ax.imshow(weights)

        logger.add_figure(title, fig, global_step=global_step)

        plt.close(fig)
    elif att_style in {"dual", "fused"}:
        # Partner
        # ===================================================
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.set_tight_layout(True)

        if spk is not None:
            timestep_mask = spk_side[1:] == spk
            history_mask = spk_side != spk  # Partner mask
        else:
            timestep_mask = np.ones_like(spk_side[1:], dtype=np.bool)
            history_mask = np.ones_like(spk_side, dtype=np.bool)

        weights_p = weights[0, timestep_mask]
        weights_p = np.array([x[history_mask] for x in np.unstack(weights_p, axis=0)])
        weights_p = (weights_p - weights_p.min(0)) / (
            ((weights_p.max(0) - weights_p.min(0))) + 1e-10
        )
        ax.imshow(weights_p)
        logger.add_figure(f"{title} (partner)", fig, global_step=global_step)
        plt.close(fig)

        # Self
        # ===================================================
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.set_tight_layout(True)

        if spk is not None:
            timestep_mask = spk_side[1:] == spk
            history_mask = spk_side == spk  # Self
        else:
            timestep_mask = np.ones_like(spk_side[1:], dtype=np.bool)
            history_mask = np.ones_like(spk_side, dtype=np.bool)

        weights_s = weights[1, timestep_mask]
        weights_s = np.array([x[history_mask] for x in np.unstack(weights_s, axis=0)])
        weights_s = (weights_s - weights_s.min(0)) / (
            ((weights_s.max(0) - weights_s.min(0))) + 1e-10
        )
        ax.imshow(weights_s)
        logger.add_figure(f"{title} (self)", fig, global_step=global_step)
        plt.close(fig)
    else:
        raise Exception(f"Unknown attention style {att_style}")
