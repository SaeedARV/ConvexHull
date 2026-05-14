from __future__ import annotations

from .python_impl import PythonBackend
from .numba_impl import NumbaBackend, nb
from .cpp_impl import CppBackend, convex_hull_cpp


def get_backend(name: str):
    name = name.lower()
    if name == "auto":
        if convex_hull_cpp is not None:
            return CppBackend()
        if nb is not None:
            return NumbaBackend()
        return PythonBackend()
    if name == "python":
        return PythonBackend()
    if name == "numpy":
        return PythonBackend()
    if name == "numba":
        return NumbaBackend()
    if name == "cpp":
        return CppBackend()
    raise ValueError(f"Unknown backend: {name}")
