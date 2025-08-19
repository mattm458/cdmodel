from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt


def plot_weights(
    weights: ArrayLike,
    speaker_designation: ArrayLike,
    speaker: int,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_tight_layout(True),

    timestep_mask = speaker_designation[1:] == speaker
    history_mask = (speaker_designation != speaker) | (speaker_designation == 0)

    weights = weights[timestep_mask]
    weights = np.array([x[history_mask] for x in np.unstack(weights, axis=0)])
    weights = (weights - weights.min(0)) / (((weights.max(0) - weights.min(0))) + 1e-10)

    ax.imshow(weights)

    return fig
