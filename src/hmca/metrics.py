"""Per-scene six-mode forecasting metrics on preprocessed arrays."""

import numpy as np


def score(trajectories, scores, target, valid):
    """Score six [T, 2] modes. True marks a valid target timestep.

    The final evaluation timestep must be valid. Scores can be logits or
    probabilities because only their ordering is used.
    """
    trajectories = np.asarray(trajectories)
    scores = np.asarray(scores)
    target = np.asarray(target)
    valid = np.asarray(valid, dtype=bool)
    if trajectories.shape != (6, *target.shape) or scores.shape != (6,):
        raise ValueError("Expected six trajectory modes and six scalar scores.")
    if target.ndim != 2 or target.shape[1] != 2 or valid.shape != target.shape[:1]:
        raise ValueError("Expected target [T, 2] and validity [T].")
    if not len(valid) or not valid[-1]:
        raise ValueError("The final target timestep must be valid.")
    displacement = np.linalg.norm(trajectories[:, valid] - target[valid], axis=-1)
    fde = displacement[:, -1]
    ade = displacement.mean(-1)
    top = int(scores.argmax())
    return {
        "minFDE6": float(fde.min()),
        "minADE6": float(ade.min()),
        "top1_FDE": float(fde[top]),
        "top1_ADE": float(ade[top]),
        "MR": float(fde.min() > 2),
    }
