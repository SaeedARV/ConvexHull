# ConvexHull
Fast and scalable randomized approximation algorithms for convex hulls, plus a
deep learning baseline.

## Install locally

Install the package in editable mode before running scripts directly:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a `uv`-managed environment:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

## Repository layout

- `src/convex_hull/`: reusable package code.
- `src/convex_hull/approximation/`: shared ApproxConvexHull core with Python, Numba, and C++ backends.
- `src/convex_hull/solvers/`: compatibility wrappers for Extents/MVEE plus DeepHull.
- `src/convex_hull/geometry/`, `sampling/`, `utils/`, `visualization/`: shared support code.
- `experiments/`: experiment runner, checked-in outputs, and small model artifacts.
- `scripts/`: standalone benchmark/demo entrypoints.
- `notebooks/`: exploratory notebooks.
- `tests/`: import and solver smoke tests.
- `paper/`: local paper sources, ignored by Git.

## Run experiments via Docker

Build:

```bash
docker build -t convexhull-experiments .
```

You may configure a Docker registry mirror on the server for pulling the base
image, then pass separate Debian and PyPI mirrors for `apt-get` and `pip`:

```bash
docker build -t convexhull-experiments \
  --build-arg APT_DEBIAN_MIRROR=https://mirror.example.ir/debian \
  --build-arg APT_SECURITY_MIRROR=https://mirror.example.ir/debian-security \
  --build-arg PIP_INDEX_URL=https://pypi.example.ir/simple \
  .
```

For an HTTP PyPI mirror or a mirror with a certificate setup that pip does not
trust, also pass its host name:

```bash
docker build -t convexhull-experiments \
  --build-arg APT_DEBIAN_MIRROR=http://mirror.example.ir/debian \
  --build-arg APT_SECURITY_MIRROR=http://mirror.example.ir/debian-security \
  --build-arg PIP_INDEX_URL=http://pypi.example.ir/simple \
  --build-arg PIP_TRUSTED_HOST=pypi.example.ir \
  .
```

The Docker registry mirror only helps Docker pull images such as
`python:3.11-slim-bookworm`; it does not mirror Debian `apt` packages or Python
packages from PyPI. If you do not pass these build arguments, Docker uses the
standard Debian and PyPI sources.

Run and write results to your local `outputs/` directory:

```bash
mkdir -p experiments/outputs
docker run --rm \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)/experiments/outputs:/app/experiments/outputs" \
  convexhull-experiments
```

## Available solvers
- `ApproxConvexHull` — preferred API for uniform and MVEE-guided approximate hulls.
- `ConvexHullviaExtents` — compatibility wrapper for uniform directional sampling.
- `ConvexHullviaMVEE` — compatibility wrapper for MVEE-guided vMF sampling.
- `ConvexHullviaDeepHull` — DeepHull with selectable backends: the original ICNN
  or a smooth convex Lipschitz ICNN.

```python
import numpy as np
from convex_hull import ApproxConvexHull

points = np.random.default_rng(0).standard_normal(size=(500, 3))
hull = ApproxConvexHull(points, method="uniform", m=2048, backend="auto", random_state=0)
approx_points = hull.get_vertices_points()

hull_mvee = ApproxConvexHull(points, method="mvee_vmf", m=16, kappa=25.0, backend="auto")
```

`backend="auto"` chooses the C++ extension when installed, then Numba, then
the vectorized Python backend.

The DeepHull solver trains a convex function through an adversarial
classification loss, producing a tight convex hull approximation directly from
the supplied points. Install PyTorch to enable this method; the implementation
automatically falls back to CPU when GPU acceleration is not available. Choose
between the baseline ICNN (`method="original"`) and the learning-theoretic
convex variant (`method="convex"`) via the `method` argument.

```python
import numpy as np
from convex_hull import ConvexHullviaDeepHull

points = np.random.rand(30, 3)
deephull = ConvexHullviaDeepHull(max_epochs=150, method="convex", lipschitz_constant=1.0)
vertex_indices = deephull.predict_indices(points)
approx_points = points[vertex_indices]
```

## Benchmarking the methods

Use `scripts/benchmark_convex_hulls.py` to generate synthetic datasets (uniform, rotated
Gaussian, and annulus shells), run all solvers, and report accuracy/runtime
metrics against SciPy’s exact hull:

```bash
python3 scripts/benchmark_convex_hulls.py --datasets uniform,gaussian,annulus \
    --samples 200 --min-points 50 --max-points 50 --dimension 5
```

Key columns in the output table:

- `Vertex F1` / `Precision` / `Recall`: agreement with the exact hull vertices.
- `Area ratio`: predicted hull area divided by the exact hull area.
- `Coverage`: fraction of all input points lying inside the predicted hull.
- `Vertices`: average number of vertices returned by the solver.
- `Runtime (ms)`: wall-clock time per instance.

DeepHull results appear automatically when PyTorch is installed. Additional
training controls are exposed via command-line flags such as
`--deephull-epochs`, `--deephull-lambda`, `--deephull-epsilon`,
`--deephull-method`, `--deephull-lipschitz`, and `--dimension`
for higher-dimensional benchmarks. Use `--deephull-method both` (or a
comma-separated list like `original,convex`) to benchmark both DeepHull variants in a
single run.
