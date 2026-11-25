"""The frozen HMCA geometry path for a single preprocessed scene."""

import numpy as np

from ._alignment import fixed_prediction, observations
from .geometry import Geometry, refine, wrap


FUSION_WEIGHTS = (
    (0.397699548937307, 0.3621925683344833, 0.2401078827282096),
    (0.49774527526749746, 0.5022412858260525, 1.3438906449964724e-05),
)


def refine_scene(inputs, initial, shrink=FUSION_WEIGHTS):
    """Refine and fuse a forward pose [tx, ty, yaw] in meters and radians.

    Inputs are actor positions [A, 50, 2], headings [A, 50], actor padding
    [A, 50], lane points [L, P, 2] and lane padding [L, P]. True marks padding.
    """
    positions, headings, padding, lanes, lane_padding = inputs
    xy, yaw, weights, _, _ = observations(positions, headings, padding)
    geometry = Geometry(xy, yaw, weights, lanes, lane_padding)
    proposal, diagnostics = refine(geometry, initial, True)
    return fixed_prediction(proposal, initial, np.asarray(shrink)), diagnostics


def inverse_map(points, headings, forward_pose):
    """Invert a forward pose on map points and headings in float64."""
    tx, ty, theta = np.asarray(forward_pose, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    inverse_row_rotation = np.array([[c, -s], [s, c]])
    corrected = (np.asarray(points, dtype=np.float64) - [tx, ty]) @ inverse_row_rotation
    return corrected, wrap(np.asarray(headings, dtype=np.float64) - theta)
