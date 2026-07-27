"""Set LD_LIBRARY_PATH and preload nvjitlink for bitsandbytes / Unsloth 4-bit on Colab."""

from __future__ import annotations

import ctypes
import glob
import os
import site


def _find_nvjitlink_libs() -> list[str]:
    libs: list[str] = []
    for sp in site.getsitepackages():
        pattern = os.path.join(sp, "nvidia", "nvjitlink", "lib", "libnvJitLink.so*")
        libs.extend(sorted(glob.glob(pattern)))
    return libs


def _preload_shared_libs() -> None:
    """Load nvjitlink into the process (works even if LD_LIBRARY_PATH was set late)."""
    for lib_path in _find_nvjitlink_libs():
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def setup_cuda_libs() -> list[str]:
    """Add nvidia pip package libs to LD_LIBRARY_PATH before importing bitsandbytes."""
    candidates: list[str] = []
    for sp in site.getsitepackages():
        for sub in (
            "nvidia/nvjitlink/lib",
            "nvidia/cuda_runtime/lib",
            "nvidia/cublas/lib",
            "nvidia/cudnn/lib",
            "nvidia/cusparse/lib",
            "nvidia/cusolver/lib",
        ):
            path = os.path.join(sp, sub)
            if os.path.isdir(path):
                candidates.append(path)
        candidates.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))

    for path in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib"):
        if os.path.isdir(path):
            candidates.append(path)

    seen: set[str] = set()
    unique: list[str] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)

    if unique:
        prefix = ":".join(unique)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{current}" if current else prefix

    _preload_shared_libs()
    return unique


CUDA_LIB_PATHS = setup_cuda_libs()
