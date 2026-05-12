"""Tests for :mod:`src.utils.device` and :mod:`src.utils.seeding`.

These two modules are foundational and must actually work (not be stubs). All other agents
depend on them.
"""

from __future__ import annotations

import os

import pytest


def test_get_device_returns_torch_device() -> None:
    pytest.importorskip("torch")
    from src.utils.device import get_device

    dev = get_device()
    assert str(dev) in {"cuda", "mps", "cpu"}


def test_get_device_respects_force_env(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from src.utils.device import get_device

    monkeypatch.setenv("CELLPATH_FORCE_DEVICE", "cpu")
    dev = get_device()
    assert str(dev) == "cpu"


def test_set_seed_makes_torch_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from src.utils.seeding import set_seed

    set_seed(7)
    a = torch.rand(3)
    set_seed(7)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_deterministic() -> None:
    import numpy as np

    from src.utils.seeding import set_seed

    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)
    assert (a == b).all()
