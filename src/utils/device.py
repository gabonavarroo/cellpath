"""Device detection — the ONLY module that may call :func:`torch.device`.

Owner: shared (Agents A + B coordinate on changes).

Resolution precedence
---------------------
1. Environment variable ``CELLPATH_FORCE_DEVICE`` (used by Docker images and CI).
2. CUDA, if available.
3. Apple MPS, if available and built.
4. CPU.

Why centralized: pinning device detection in one place avoids drift between scVI's internal
heuristics, ``stable-baselines3``'s auto-detect, and ad-hoc training loops. Both agents call
:func:`get_device` and pass the result downstream.
"""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Return the best available accelerator.

    Returns
    -------
    torch.device
        ``cuda`` / ``mps`` / ``cpu`` depending on availability. Can be forced via
        the ``CELLPATH_FORCE_DEVICE`` environment variable.

    Examples
    --------
    >>> dev = get_device()
    >>> str(dev) in {"cuda", "mps", "cpu"}
    True
    """
    forced = os.environ.get("CELLPATH_FORCE_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary() -> str:
    """Human-readable string describing the active device + version info.

    Returns
    -------
    str
        Single-line summary suitable for logging at program start.
    """
    dev = get_device()
    parts = [f"device={dev}", f"torch={torch.__version__}"]
    if dev.type == "cuda":
        parts.append(f"cuda={torch.version.cuda}")
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    return " | ".join(parts)


if __name__ == "__main__":
    # Used as a quick sanity check: `python -m src.utils.device`
    print(device_summary())
