import json
from pathlib import Path
import unittest

import numpy as np

from hmca.__main__ import synthetic_scene
from hmca.core import FUSION_WEIGHTS, inverse_map, refine_scene
from hmca.geometry import synthetic
from hmca.metrics import score
from hmca.ralr import Solver
from hmca.se2 import exp_se2, log_se2


class AlignmentTests(unittest.TestCase):
    def test_se2_round_trip(self):
        poses = np.array([[.5, -.25, 0.], [2., .7, 1e-9], [1., 2., -.04]])
        np.testing.assert_allclose(exp_se2(log_se2(poses)), poses, atol=1e-12, rtol=0)

    def test_synthetic_map_and_refinement(self):
        inputs, initial, injected = synthetic_scene()
        points, headings = inverse_map(inputs[3], inputs[1] + injected[2], injected)
        np.testing.assert_allclose(points, inputs[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(headings, inputs[1], atol=1e-12, rtol=0)
        pose, report = refine_scene(inputs, initial)
        self.assertGreater(report["accepted_steps"], 0)
        self.assertLess(np.linalg.norm(pose - injected), np.linalg.norm(initial - injected))
        self.assertLess(abs(synthetic()["longitudinal_cost_change"]), 1e-12)

    def test_frozen_protocol_and_ralr(self):
        config = Path(__file__).resolve().parents[1] / "configs"
        protocol = json.loads((config / "protocol.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(protocol["fusion"]["weights"], FUSION_WEIGHTS)
        r = Solver(json.loads((config / "ralr.json").read_text(encoding="utf-8")))
        (p, h, pad, lanes, lane_pad), _, _ = synthetic_scene()
        pose, info = r.solve(p[0], h[0], pad[0], p, h, pad, lanes, lane_pad, lane_pad.all(axis=1))
        self.assertTrue(np.isfinite(pose).all())
        self.assertTrue(info["success"])

    def test_minade_and_minfde_are_independent(self):
        modes = np.full((6, 2, 2), 10.)
        modes[0] = [[5., 0.], [2., 0.]]
        modes[1] = [[0., 0.], [3., 0.]]
        result = score(modes, [1., 3., 0., 0., 0., 0.], np.zeros((2, 2)), [True, True])
        self.assertEqual(result, dict(minFDE6=2., minADE6=1.5, top1_FDE=3., top1_ADE=1.5, MR=0.))
