# ConvexHull
Fast and scalable randomized approximation algorithms for convex hulls, plus a
deep learning baseline.

## Available solvers
- `ConvexHullviaExtents` — deterministic sampling of directional extents.
- `ConvexHullviaMVEE` — MVEE-based approximation.
- `ConvexHullviaDeepHull` — transformer model distilled from Meta's DeepHullNet.

The DeepHull solver lives in `ConvexHullviaDeepHull.py` and loads the pretrained
weights stored in `deephull_transform_best_params.pkl`.  Install PyTorch before
using this method.  The helper auto-selects CUDA or Apple MPS GPUs when available.

```python
import numpy as np
from ConvexHullviaDeepHull import ConvexHullviaDeepHull

points = np.random.rand(30, 2)
deephull = ConvexHullviaDeepHull()
predicted_vertices = deephull.compute(points)
```

## Benchmarking the methods

Use `benchmark_convex_hulls.py` to generate synthetic datasets (uniform, rotated
Gaussian, and annulus shells), run all solvers, and report accuracy/runtime
metrics against SciPy’s exact hull:

```bash
python benchmark_convex_hulls.py --datasets uniform,gaussian,annulus \
    --samples 200 --min-points 20 --max-points 50
```

Key columns in the output table:

- `Vertex F1` / `Precision` / `Recall`: agreement with the exact hull vertices.
- `Area ratio`: predicted hull area divided by the exact hull area.
- `Coverage`: fraction of all input points lying inside the predicted hull.
- `Vertices`: average number of vertices returned by the solver.
- `Runtime (ms)`: wall-clock time per instance.

DeepHull results appear automatically when PyTorch and the checkpoint file are available.
