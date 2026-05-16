"""P0B' — pairing comparator + Contract-2 schema regression tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


CONTRACT2_KEYS_TRAIN = {"z_ctrl", "gene_idx", "z_pert"}
CONTRACT2_KEYS_COMBO = {"z_ctrl", "gene_idx_a", "gene_idx_b", "z_pert_ab"}


def _schema_ok(npz_path: Path, keys: set[str], n_latent: int = 32) -> None:
    data = np.load(npz_path)
    assert set(data.files) >= keys, f"{npz_path} missing keys: {keys - set(data.files)}"
    if "z_ctrl" in data.files:
        assert data["z_ctrl"].shape[1] == n_latent


@pytest.mark.parametrize("subdir", ["pairs_mean_delta", "pairs_random"])
def test_pairs_schema_contract2(subdir: str) -> None:
    root = Path("artifacts_v2") / subdir
    if not root.exists():
        pytest.skip(f"{root} not produced yet — runs before Task 4+ are expected to skip")
    for fname, keys in [
        ("train_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("val_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("ood_pairs.npz", CONTRACT2_KEYS_TRAIN),
        ("combo_pairs.npz", CONTRACT2_KEYS_COMBO),
    ]:
        _schema_ok(root / fname, keys)
    assert (root / "metadata.json").exists()


def test_v1_pairs_unchanged() -> None:
    """Sanity check: V1 OT pair metadata is byte-identical to its session-start SHA.

    Guards against accidental writes to artifacts/pairs/ during P0B' runs. The
    expected SHA is read from a sibling fixture file maintained by the executor
    on the first P0B' run; if absent, the test seeds it.
    """
    target = Path("artifacts/pairs/metadata.json")
    if not target.exists():
        pytest.skip("artifacts/pairs/metadata.json not present")
    fixture = Path("tests/fixtures/v1_pairs_metadata.sha256")
    h = hashlib.sha256(target.read_bytes()).hexdigest()
    if not fixture.exists():
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(h)
        pytest.skip("seeded fixture; rerun")
    assert fixture.read_text().strip() == h, "V1 OT pairs metadata changed — investigate"


def test_pairing_comparison_emits_required_keys(tmp_path: Path) -> None:
    """Comparator must read gate.json + pairing_noise.json and emit a tidy row schema."""
    from scripts import compare_pairings  # fails hard if Task 3 not yet implemented

    fake_runs = []
    for name in ("ot", "mean_delta", "random"):
        pdir = tmp_path / f"pairs_{name}"
        ddir = tmp_path / f"dynamics_{name}"
        pdir.mkdir()
        ddir.mkdir()
        (pdir / "metadata.json").write_text(
            json.dumps({"n_train": 1000, "pairing_method": name})
        )
        (ddir / "gate.json").write_text(json.dumps({
            "passed": False,
            "primary": {
                "pearson_r": 0.5,
                "baselines": {"linear_ridge": {"pearson_r": 0.49}},
            },
            "ood": {
                "pearson_r": 0.45,
                "baselines": {"linear_ridge": {"pearson_r": 0.43}},
            },
            "uncertainty_calibration": {"spearman": 0.22},
        }))
        (ddir / "pairing_noise.json").write_text(
            json.dumps({"summary": {"median_noise_ratio": 0.7}})
        )
        fake_runs.append((name, str(pdir), str(ddir)))

    out = tmp_path / "pairing_comparison"
    compare_pairings.main(runs=fake_runs, out=str(out))

    record = json.loads(Path(str(out) + ".json").read_text())
    required = {
        "pairing_method",
        "n_train",
        "pairing_noise_median",
        "val_mlp_minus_ridge_pearson",
        "ood_mlp_minus_ridge_pearson",
        "uncertainty_spearman",
        "gate_passed",
    }
    assert set(record["rows"][0].keys()) >= required
