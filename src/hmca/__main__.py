"""Run HMCA on a synthetic example or a compatible preprocessed scene."""

import argparse
import json
from pathlib import Path

import numpy as np

from .core import refine_scene


FIELDS = ("positions", "headings", "padding", "lanes", "lane_padding")


def synthetic_scene():
    """Two orthogonal lanes and moving, aligned actor histories."""
    t = np.linspace(-15, 10, 50)
    positions = np.stack((
        np.column_stack((t, np.zeros_like(t))),
        np.column_stack((np.zeros_like(t), t)),
    ))
    headings = np.stack((np.zeros_like(t), np.full_like(t, np.pi / 2)))
    theta = np.deg2rad(2.)
    c, s = np.cos(theta), np.sin(theta)
    injected = np.array([1., .75, theta])
    lanes = positions @ np.array([[c, s], [-s, c]]) + injected[:2]
    inputs = (
        positions, headings, np.zeros(headings.shape, dtype=bool),
        lanes, np.zeros(headings.shape, dtype=bool),
    )
    return inputs, np.array([.3, .2, 0.]), injected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="NPZ with preprocessed inputs and an initial pose.")
    options = parser.parse_args()
    known_pose = None
    if options.scene:
        with np.load(options.scene, allow_pickle=False) as data:
            inputs = tuple(data[name] for name in FIELDS)
            initial = data["initial_pose"]
    else:
        inputs, initial, known_pose = synthetic_scene()
    pose, diagnostics = refine_scene(inputs, initial)
    result = {
        "initial_pose_m_rad": initial.tolist(),
        "fused_pose_m_rad": pose.tolist(),
        "diagnostics": diagnostics,
    }
    if known_pose is not None:
        result["synthetic_reference_pose_m_rad"] = known_pose.tolist()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
