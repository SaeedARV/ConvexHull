# ConvexHull
Fast and scalable randomized approximation algorithms for convex hulls, plus a
deep learning baseline.

## Available solvers
- `ConvexHullviaExtents` — deterministic sampling of directional extents.
- `ConvexHullviaMVEE` — MVEE-based approximation.
- `ConvexHullviaDeepHull` — input-convex neural network trained on demand following
  Balestriero et al.'s DeepHull formulation.

The DeepHull solver trains a small input-convex network (ICNN) through an
adversarial classification loss, producing a tight convex hull approximation
directly from the supplied points. Install PyTorch to enable this method; the
implementation automatically falls back to CPU when GPU acceleration is not
available.

```python
import numpy as np
from ConvexHullviaDeepHull import ConvexHullviaDeepHull

points = np.random.rand(30, 3)
deephull = ConvexHullviaDeepHull(max_epochs=150)
vertex_indices = deephull.predict_indices(points)
approx_points = points[vertex_indices]
```

## Benchmarking the methods

Use `benchmark_convex_hulls.py` to generate synthetic datasets (uniform, rotated
Gaussian, and annulus shells), run all solvers, and report accuracy/runtime
metrics against SciPy’s exact hull:

```bash
python3 benchmark_convex_hulls.py --datasets uniform,gaussian,annulus \
    --samples 200 --min-points 50 --max-points 50
```

Key columns in the output table:

- `Vertex F1` / `Precision` / `Recall`: agreement with the exact hull vertices.
- `Area ratio`: predicted hull area divided by the exact hull area.
- `Coverage`: fraction of all input points lying inside the predicted hull.
- `Vertices`: average number of vertices returned by the solver.
- `Runtime (ms)`: wall-clock time per instance.

DeepHull results appear automatically when PyTorch is installed. Additional
training controls are exposed via command-line flags such as
`--deephull-epochs`, `--deephull-lambda`, and `--deephull-epsilon`.
