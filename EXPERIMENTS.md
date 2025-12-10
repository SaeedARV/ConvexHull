# Experiment Guide

This repository ships a single script, `run_experiments.py`, that runs all convex-hull experiments end-to-end. The script uses the provided solvers (`ConvexHullviaExtents`, `ConvexHullviaMVEE`, `ConvexHullviaDeepHull`) and saves numeric outputs plus publication-style plots. You can run everything with:

```bash
python run_experiments.py --output-dir outputs
```

Use `--help` to see optional flags (seed, DeepHull hyperparameters, output directory).

## What the script does

`run_experiments.py` conducts three suites:

1) **Low-dimensional exact regime (d ∈ {2, 3, 5, 8}, n ∈ {200, 500, 1000})**  
   - Datasets: uniform cube, isotropic Gaussian, and anisotropic Gaussians (condition numbers 1, 10, 100).  
   - Reference: exact SciPy `ConvexHull`.  
   - Metrics per method: support-function error (≈2000 random directions), Monte-Carlo volume ratio, membership accuracy (inside/outside), runtime, and number of vertices.  
   - Plots: support error vs n, volume ratio vs n, runtime vs n (log-scale), and 2D/3D visualizations for representative cases.

2) **High-dimensional scalability (d ∈ {50, 100, 200, 500}, n ∈ {2000, 5000, 10000})**  
   - Datasets: Gaussian and uniform cube.  
   - Exact Qhull is attempted once with a short timeout and otherwise skipped.  
   - Metrics per method: runtime, memory footprint of vertex sets, Qhull feasibility flag. Pairwise support-function discrepancies are collected across random directions.  
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

## What gets saved

All artifacts live under the chosen `--output-dir` (default `outputs/`):

- `low_dim/low_dim_results.csv`: full metric table for the exact regime.  
- `low_dim/*.png`: support/volume/runtime curves and 2D/3D visualizations.  
- `high_dim/high_dim_runtime.csv`: runtimes, memory, status flags.  
- `high_dim/high_dim_support_pairs.csv`: pairwise support-function differences.  
- `high_dim/*.png`: runtime curves, feasibility heatmap, violin plot.  
- `anomaly/anomaly_detection.csv`: AUROC/AUPR summary.  
- `anomaly/*.png`: ROC, PR, and summary bar plots.  
- `manifest.json`: index of all generated files for convenience.

If PyTorch or `ConvexHullviaDeepHull` is unavailable, the DeepHull rows/plots are skipped gracefully; Extents and MVEE still run.
