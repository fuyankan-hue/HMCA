# HMCA

HMCA aligns observed lane geometry to actor histories before motion forecasting. A three-branch initializer estimates the forward scene-level SE(2) offset. Dynamic actor–lane correspondence, robust nonlinear refinement, and fixed logarithmic fusion refine that estimate. The inverse transformation corrects the map upstream of a frozen predictor.

The public package provides the standalone alignment core, direct initializer and pose loss, RALR comparison solver, archived validation aggregates, and a synthetic CPU example. It starts from preprocessed tensors. The archived EMP-D and QCNet results were produced with backbone-specific preprocessing and separately trained initialization heads. Their complete evaluation pipelines, dataset files, and checkpoints are outside the scope of this release.

## Run the alignment example

From the repository root, with Python 3.10 or later:

    python -m pip install .
    python -m hmca
    python -m unittest discover -s tests -p test_geometry.py

The demonstration creates two perpendicular lanes in memory and prints the synthetic injected pose, initializer, fused pose, and diagnostics. It requires no dataset, weights, or GPU.

## Use a preprocessed scene

    python -m hmca --scene scene.npz

The NPZ contains the following arrays in one common focal-centered frame:

| Key | Shape | Meaning |
| --- | --- | --- |
| positions | [A, 50, 2] | Observed actor positions in meters |
| headings | [A, 50] | Observed actor headings in radians |
| padding | [A, 50] | Boolean actor mask, True = padding |
| lanes | [L, P, 2] | Observed lane polyline sample points in meters |
| lane_padding | [L, P] | Boolean lane-point mask, True = padding |
| initial_pose | [3] | Forward pose estimate [tx, ty, yaw], in meters and radians |

Use finite fill values for padding. The inference inputs use observed histories and lane geometry. Call hmca.core.refine_scene from Python for the pose and diagnostics. Apply hmca.core.inverse_map to the observed map points and headings using the fused forward pose. Keep actor histories fixed.

Install the optional initializer dependency with:

    python -m pip install ".[encoder]"

hmca.initializer.DirectInitializer consumes the focal and all-actor histories separately, with batched versions of the same history and map arrays. It emits a bounded forward pose. hmca.initializer.direct_pose_loss implements the supervised translation and scaled-yaw Smooth L1 objective. The default initializer starts with untrained weights. The archived checkpoint fingerprints in configs/protocol.json identify the weights used in the study.

## Reproduction files

| File | Contents |
| --- | --- |
| configs/protocol.json | Frozen fusion weights, training schedule, validation conditions, seed rules, bootstrap settings and archival SHA256 values |
| configs/ralr.json | Frozen RALR solver parameters, used with hmca.ralr.Solver |
| results/empd_validation.csv | EMP-D aggregate AV2 validation metrics for 16 conditions and four methods |
| results/qcnet_validation.csv | QCNet aggregate AV2 validation metrics for 10 conditions and four methods |
| results/paired_forecast_effects.csv | HMCA paired minFDE6 differences and scene-bootstrap intervals |
| results/component_ablation.csv | Matched component comparisons on held-out AV2 Training cohorts |
| results/translation_distribution.csv | Summary statistics of formal translation residuals |

The CSVs report the archived numerical values. MR in the validation CSVs is a fraction. Columns ending in _pct are percentages. For EMP-D, Clean is scored once per scene and reused across the three perturbation seeds for reporting. Its raw counts in the aggregate CSV count scene–seed records, while the manuscript's Clean tail table counts unique scenes. QCNet uses perturbation seed 2333 only. Each backbone uses a separately trained initialization head with training seed 2333. Confidence intervals resample scenarios with perturbation replicates grouped within each draw.

## Upstream projects

- [Efficient Motion Prediction](https://github.com/a-pru/emp)
- [QCNet](https://github.com/ZikangZhou/QCNet)
- [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html)

Obtain AV2 and backbone assets through their upstream projects under the applicable terms. This repository contains only aggregate benchmark results and synthetic example inputs generated in memory.

## License

The source code is released under the [BSD 3-Clause License](LICENSE). The original EMP notices and terms are retained in [LICENSES/EMP-BSD-3-Clause.txt](LICENSES/EMP-BSD-3-Clause.txt).
