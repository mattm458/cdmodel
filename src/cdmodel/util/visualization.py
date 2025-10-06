import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from cdmodel.model.types import AttentionMaskingStrategy


def plot_weights(
    weights: ArrayLike,
    masking_strategy: AttentionMaskingStrategy,
    spk_side: ArrayLike,
    spk: int | None,
    title: str,
    logger,
    global_step: int,
):
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_tight_layout(True)

    if spk is not None:
        timestep_mask = spk_side[1:] == spk
        if masking_strategy == "partner":
            history_mask = spk_side != spk
        elif masking_strategy == "both":
            history_mask = np.ones_like(spk_side, dtype=np.bool)
    else:
        timestep_mask = np.ones_like(spk_side[1:], dtype=np.bool)
        history_mask = np.ones_like(spk_side, dtype=np.bool)

    weights = weights[timestep_mask]
    weights = np.array([x[history_mask] for x in np.unstack(weights, axis=0)])
    weights = (weights - weights.min(0)) / (((weights.max(0) - weights.min(0))) + 1e-10)

    ax.imshow(weights)

    logger.add_figure(title, fig, global_step=global_step)

    plt.close(fig)
