# Experiment Guide

This repository ships a single script, `experiments/run_experiments.py`, that runs all convex-hull experiments end-to-end. The script uses the provided solvers (`ApproxConvexHull` through the Extents/MVEE wrappers, plus `ConvexHullviaDeepHull`) and saves numeric outputs plus publication-style plots. You can run everything with:

```bash
python experiments/run_experiments.py --output-dir experiments/outputs
```

If you run from the `experiments/` directory instead, use `--output-dir outputs`.

Use `--help` to see optional flags (seed, DeepHull hyperparameters, output directory, steps).

## Command-line arguments

- `--output-dir` (default: `outputs`): root directory for results and plots.
- `--seed` (default: `7`): base random seed for all experiments.
- `--steps` (default: `low,high,anomaly`): comma-separated list of stages to run.
- `--high-extents-samples` (default: `2048`): number of random directions for Extents in high-dimensional runs.
- `--deephull-epochs` (default: `150`): training epochs for DeepHull.
- `--deephull-lambda` (default: `2.0`): negative sample weight for DeepHull.
- `--deephull-epsilon` (default: `0.05`): boundary tolerance for DeepHull.
- `--deephull-lipschitz` (default: `1.0`): Lipschitz constant for convex ICNN.
- `--log-file` (default: `output-dir/run_experiments_YYYYmmdd_HHMMSS.txt`): write progress logs to this file.
- `--no-console-log`: disable console logging (log file still written).
- `--resume`: resume from existing CSVs and append new rows.

## What the script does

`run_experiments.py` conducts three suites:

1) **Low-dimensional exact regime (d ∈ {2, 3, 5, 8}, n ∈ {200, 500, 1000})**  
   - Datasets: uniform cube, isotropic Gaussian, and anisotropic Gaussians (condition numbers 1, 10, 100).  
   - Reference: exact SciPy `ConvexHull`.  
   - Metrics per method: support-function error (≈2000 random directions), Monte-Carlo volume ratio, membership accuracy (inside/outside), runtime, and number of vertices.  
   - Plots: support error vs n, volume ratio vs n, runtime vs n (log-scale), and 2D/3D visualizations for representative cases.

2) **High-dimensional scalability (d ∈ {50, 100, 200, 500}, n ∈ {2000, 5000, 10000})**  
   - Datasets: Gaussian and uniform cube.  
   - Exact Qhull is skipped (it is infeasible at these dimensions).  
   - Metrics per method: runtime and memory footprint of vertex sets. Pairwise support-function discrepancies are collected across random directions.  
   - Plots: runtime vs n (per dimension), runtime vs dimension (per n), feasibility heatmap (finish within 60s), and violin plots of pairwise support distances.

3) **Downstream anomaly detection (OOD)**  
   - Data: two 10D Gaussians; one is in-distribution (10k points), the other OOD (5k points).  
   - Scoring: support-function boundary distance on each trained hull.  
   - Metrics per method: AUROC, AUPR, runtime, vertex count.  
   - Plots: ROC curves, PR curves, and a bar chart summarizing AUROC/AUPR.

## Why these experiments

- **Low-dimensional exact regime**: stresses geometric fidelity where ground truth hulls are available. Support errors, volume ratios, and membership accuracy directly reveal approximation quality of Extents/MVEE/DeepHull without confounding high-d limits.  
- **High-dimensional scalability**: targets the methods’ intended use case—large d where exact hulls are infeasible. Runtime, memory, and feasibility probe whether each representation remains practical; pairwise support comparisons show relative geometric agreement when no ground truth exists.  
- **Anomaly detection**: validates that hull quality transfers to a downstream task. Support-function margins are a natural scoring rule for all three solvers, so AUROC/AUPR illustrate how geometric differences affect end-user performance.

## Metric definitions (CSV columns)

### `low_dim/low_dim_results.csv`

- `dimension`, `n_points`, `dataset`, `method`: identifiers for the run (dataset is `uniform_cube`, `gaussian`, or `anisotropic_cond{1,10,100}`; method is `Extents`, `MVEE`, `DeepHull-original`, `DeepHull-convex`).
- `support_error`: max absolute support-function discrepancy over ~2000 random unit directions. Support is `h_K(u) = max_{x in K} u^T x`; lower is better.
- `volume_ratio`: Monte Carlo estimate of `vol(approx) / vol(true)` using 5000 random samples in the true hull's bounding box; closer to 1 is better.
- `volume_true`, `volume_approx`: the raw Monte Carlo volume estimates used to compute `volume_ratio`.
- `membership_acc`: fraction of 5000 random samples in the bounding box where true and approximate hulls agree on inside/outside classification; closer to 1 is better.
- `runtime_ms`: solver runtime for that method in milliseconds.
- `n_vertices`: number of vertices in the method's returned hull representation.
- `status`: `ok`, `empty` (insufficient vertices), or `missing` (method unavailable).

### `high_dim/high_dim_runtime.csv`

- `dimension`, `n_points`, `dataset`, `method`: identifiers for the run (dataset is `gaussian` or `uniform_cube`).
- `runtime_ms`: solver runtime in milliseconds.
- `memory_bytes`: `nbytes` of the vertex array returned (proxy for storage cost).
- `status`: `ok`, `empty`, or `missing`.

### `high_dim/high_dim_support_pairs.csv`

- `dimension`, `n_points`, `dataset`: identifiers for the run.
- `pair`: method pair (e.g., `Extents vs MVEE`).
- `direction`: index of the random unit direction (0-255).
- `support_diff`: absolute difference between the two methods' support-function values along that direction.

### `anomaly/anomaly_detection.csv`

- `method`: solver name.
- `auroc`: area under the ROC curve for the anomaly score (higher is better).
- `aupr`: area under the precision-recall curve for the anomaly score (higher is better).
- `runtime_ms`: solver runtime in milliseconds.
- `n_vertices`: number of vertices in the hull representation.

## What gets saved

All artifacts live under the chosen `--output-dir` (the checked-in paper outputs are under `experiments/outputs/`):

- `low_dim/low_dim_results.csv`: full metric table for the exact regime.  
- `low_dim/*.png`: support/volume/runtime curves and 2D/3D visualizations.  
- `high_dim/high_dim_runtime.csv`: runtimes, memory, status flags.  
- `high_dim/high_dim_support_pairs.csv`: pairwise support-function differences.  
- `high_dim/*.png`: runtime curves, feasibility heatmap, violin plot.  
- `anomaly/anomaly_detection.csv`: AUROC/AUPR summary.  
- `anomaly/*.png`: ROC, PR, and summary bar plots.  
- `manifest.json`: index of all generated files for convenience.

Static experiment assets that are not generated by a run live under `experiments/artifacts/`.

If PyTorch or `ConvexHullviaDeepHull` is unavailable, the DeepHull rows/plots are skipped gracefully; Extents and MVEE still run.

## Why each saved file matters for the paper

### Low-dimensional exact regime

- `low_dim/low_dim_results.csv`: the quantitative backbone for claims about geometric fidelity against ground truth. It captures accuracy (support error, volume ratio, membership) and efficiency (runtime, vertex count) so you can report tables, compute averages/variances, and run significance checks.
- `low_dim/*.png`: visual and trend evidence for the narrative. Curves show how error/volume/runtime scale with n, and 2D/3D visualisations provide intuitive, inspectable examples that corroborate the numeric metrics.

### High-dimensional scalability

- `high_dim/high_dim_runtime.csv`: supports scalability claims when exact hulls are infeasible. Runtime and memory proxy costs show practical feasibility across d and n.
- `high_dim/high_dim_support_pairs.csv`: provides a geometry-consistency signal when no ground truth exists. Pairwise support gaps quantify agreement or divergence between methods, enabling distributional comparisons beyond single summary statistics.
- `high_dim/*.png`: translates the high-d results into paper-ready figures (runtime curves, feasibility heatmap, violin plots) that emphasize scaling behavior and method trade-offs.

### Downstream anomaly detection

- `anomaly/anomaly_detection.csv`: ties geometric quality to a downstream task. AUROC/AUPR measure detection performance, while runtime and vertex count show the cost of using each method.
- `anomaly/*.png`: standard evaluation plots (ROC/PR) and summary bars that communicate method performance in a form expected by reviewers.

### Experiment index

- `manifest.json`: makes results auditable and reproducible by listing all generated artifacts, which is helpful when assembling figures or sharing outputs with collaborators.
