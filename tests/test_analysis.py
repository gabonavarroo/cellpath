"""Phase 3 tests: DepMap enrichment metrics + trajectory loading.

All tests use synthetic data — no real DepMap download required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# =============================================================================
# hypergeometric_enrichment
# =============================================================================


class TestHypergeometricEnrichment:
    def test_significant_overlap(self) -> None:
        from src.analysis.metrics import hypergeometric_enrichment

        # N=100 background, K=10 in gene set, select n=20, k=8 overlap
        # Expected overlap = 20*10/100 = 2; observing 8 is highly significant
        bg   = [f"G{i}" for i in range(100)]
        gset = bg[:10]   # K=10
        sel  = bg[:8] + bg[90:102]  # k=8 overlap, n=20 total (capped to bg)
        sel  = [g for g in sel if g in set(bg)][:20]
        result = hypergeometric_enrichment(sel, gset, bg)
        assert result["p_value"] < 0.05
        assert result["log_odds"] > 0
        assert result["N"] == 100

    def test_no_overlap(self) -> None:
        from src.analysis.metrics import hypergeometric_enrichment

        result = hypergeometric_enrichment(
            selected_genes=["X", "Y"],
            gene_set=["A", "B"],
            background=["A", "B", "X", "Y"],
        )
        assert result["k"] == 0
        assert result["p_value"] == 1.0
        assert result["log_odds"] == 0.0

    def test_empty_selected(self) -> None:
        from src.analysis.metrics import hypergeometric_enrichment

        result = hypergeometric_enrichment(
            selected_genes=[],
            gene_set=["A", "B"],
            background=["A", "B", "C"],
        )
        assert result["p_value"] == 1.0

    def test_background_filter(self) -> None:
        """Genes not in background are ignored."""
        from src.analysis.metrics import hypergeometric_enrichment

        result = hypergeometric_enrichment(
            selected_genes=["A", "PHANTOM"],
            gene_set=["A", "PHANTOM2"],
            background=["A", "B", "C"],
        )
        assert result["N"] == 3
        assert result["n"] <= 2


# =============================================================================
# gsea_preranked
# =============================================================================


class TestGseaPreranked:
    def test_positive_enrichment(self) -> None:
        from src.analysis.metrics import gsea_preranked

        rng = np.random.default_rng(0)
        genes = [f"G{i}" for i in range(50)]
        gene_set = genes[:10]  # top genes
        scores = np.arange(50, 0, -1, dtype=float) + rng.normal(0, 0.1, 50)
        result = gsea_preranked(genes, scores, gene_set, n_permutations=200, seed=0)
        assert "ES" in result and "NES" in result
        assert "p_value" in result and "fdr" in result
        assert result["ES"] > 0  # top-ranked genes should be enriched
        assert 0.0 <= result["p_value"] <= 1.0

    def test_empty_gene_set(self) -> None:
        from src.analysis.metrics import gsea_preranked

        result = gsea_preranked(
            ["A", "B", "C"], np.array([3.0, 2.0, 1.0]), [], n_permutations=10
        )
        assert result["p_value"] == 1.0
        assert result["ES"] == 0.0

    def test_result_keys(self) -> None:
        from src.analysis.metrics import gsea_preranked

        result = gsea_preranked(
            ["A", "B", "C", "D"],
            np.array([4.0, 3.0, 2.0, 1.0]),
            ["A", "B"],
            n_permutations=50,
        )
        assert set(result.keys()) == {"ES", "NES", "p_value", "fdr"}


# =============================================================================
# null_enrichment_comparison
# =============================================================================


class TestNullEnrichmentComparison:
    def test_high_enrichment_positive_z(self) -> None:
        from src.analysis.metrics import null_enrichment_comparison

        bg = [f"G{i}" for i in range(100)]
        # Observed enrichment far above null
        result = null_enrichment_comparison(
            observed_enrichment=5.0,
            selected_genes=bg[:10],
            background=bg,
            n_null_samples=200,
            seed=0,
        )
        assert result["z_score"] > 0
        assert 0.0 <= result["empirical_p"] <= 1.0
        assert "null_mean" in result and "null_std" in result

    def test_expression_matched(self) -> None:
        from src.analysis.metrics import null_enrichment_comparison

        bg = [f"G{i}" for i in range(50)]
        expr = np.linspace(0, 10, 50)
        result = null_enrichment_comparison(
            observed_enrichment=1.0,
            selected_genes=bg[:5],
            background=bg,
            n_null_samples=100,
            match_expression=expr,
            seed=0,
        )
        assert "z_score" in result

    def test_result_keys(self) -> None:
        from src.analysis.metrics import null_enrichment_comparison

        result = null_enrichment_comparison(
            observed_enrichment=0.0,
            selected_genes=["A"],
            background=["A", "B", "C"],
            n_null_samples=50,
            seed=0,
        )
        assert set(result.keys()) == {"z_score", "empirical_p", "null_mean", "null_std"}


# =============================================================================
# depmap_validation: load_depmap_k562
# =============================================================================


class TestLoadDepmapK562:
    def test_loads_real_file(self) -> None:
        from src.analysis.depmap_validation import load_depmap_k562

        real_path = Path("data/processed/depmap_k562_chronos.parquet")
        if not real_path.exists():
            pytest.skip("DepMap parquet not downloaded — run make data.")
        df = load_depmap_k562(real_path)
        assert "gene_symbol" in df.columns
        assert "chronos" in df.columns
        assert "is_essential" in df.columns
        assert len(df) > 0

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        from src.analysis.depmap_validation import load_depmap_k562

        with pytest.raises(FileNotFoundError):
            load_depmap_k562(tmp_path / "nonexistent.parquet")


# =============================================================================
# depmap_validation: load_gene_panels
# =============================================================================


class TestLoadGenePanels:
    def test_loads_txt_files(self, tmp_path: Any) -> None:
        from src.analysis.depmap_validation import load_gene_panels

        (tmp_path / "panel_a.txt").write_text("GENE1\nGENE2\n# comment\nGENE3\n")
        (tmp_path / "panel_b.txt").write_text("GENEXY\n")
        panels = load_gene_panels(tmp_path)
        assert "panel_a" in panels
        assert "panel_b" in panels
        assert panels["panel_a"] == ["GENE1", "GENE2", "GENE3"]
        assert "GENE1" not in panels["panel_b"]

    def test_missing_dir_returns_empty(self, tmp_path: Any) -> None:
        from src.analysis.depmap_validation import load_gene_panels

        panels = load_gene_panels(tmp_path / "nonexistent")
        assert panels == {}

    def test_loads_nested_panel_dirs(self, tmp_path: Any) -> None:
        from src.analysis.depmap_validation import load_gene_panels

        hallmark = tmp_path / "hallmark"
        lineage = tmp_path / "lineage"
        hallmark.mkdir()
        lineage.mkdir()
        (hallmark / "hallmark_apoptosis.txt").write_text("BAX\nBAK1\nBCL2L11\n")
        (lineage / "k562_myeloid_lineage.txt").write_text("KLF1\nSPI1\nCEBPA\n")

        panels = load_gene_panels(tmp_path)

        assert panels["hallmark_apoptosis"] == ["BAX", "BAK1", "BCL2L11"]
        assert panels["k562_myeloid_lineage"] == ["KLF1", "SPI1", "CEBPA"]

    def test_required_panel_collections_surface_missing(self, tmp_path: Any) -> None:
        from src.analysis.depmap_validation import load_gene_panel_manifest

        (tmp_path / "manifest.json").write_text(json.dumps({
            "panels": [
                {
                    "name": "hallmark_apoptosis",
                    "file": "hallmark_apoptosis.txt",
                    "collection": "hallmark",
                    "source": "synthetic",
                }
            ]
        }))

        with pytest.raises(ValueError, match="lineage"):
            load_gene_panel_manifest(tmp_path, required_collections=("hallmark", "lineage"))


# =============================================================================
# depmap_validation: run_depmap_enrichment (synthetic)
# =============================================================================


class TestRunDepmapEnrichment:
    def _mock_chronos_df(self) -> Any:
        import polars as pl

        genes = [f"G{i}" for i in range(50)]
        chronos = np.linspace(-2.0, 1.0, 50)
        return pl.DataFrame({
            "gene_symbol": genes,
            "chronos": chronos.astype(np.float32),
            "is_essential": (chronos < -0.5).tolist(),
        })

    def test_returns_dataframe(self) -> None:
        from src.analysis.depmap_validation import run_depmap_enrichment

        bg = [f"G{i}" for i in range(50)]
        freq = {f"G{i}": 50 - i for i in range(50)}
        df = run_depmap_enrichment(
            rl_action_freq=freq,
            background_genes=bg,
            panels={},
            chronos_df=self._mock_chronos_df(),
            top_k=10,
            n_null=50,
        )
        assert len(df) > 0
        assert "q_value" in df.columns
        assert "p_value" in df.columns
        assert "panel" in df.columns
        assert "test" in df.columns

    def test_writes_csv(self, tmp_path: Any) -> None:
        from src.analysis.depmap_validation import run_depmap_enrichment

        bg = [f"G{i}" for i in range(20)]
        freq = {f"G{i}": 20 - i for i in range(20)}
        out = tmp_path / "enrichment.csv"
        run_depmap_enrichment(
            rl_action_freq=freq,
            background_genes=bg,
            panels={},
            chronos_df=self._mock_chronos_df(),
            top_k=5,
            n_null=20,
            out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_q_values_between_0_and_1(self) -> None:
        from src.analysis.depmap_validation import run_depmap_enrichment

        bg = [f"G{i}" for i in range(30)]
        freq = {f"G{i}": 30 - i for i in range(30)}
        df = run_depmap_enrichment(
            rl_action_freq=freq,
            background_genes=bg,
            panels={},
            chronos_df=self._mock_chronos_df(),
            top_k=8,
            n_null=30,
        )
        q = df["q_value"].to_numpy()
        assert (q >= 0).all() and (q <= 1).all()

    def test_multi_panel_rows_include_provenance_and_null_strategy(self) -> None:
        from src.analysis.depmap_validation import run_depmap_enrichment

        bg = [f"G{i}" for i in range(50)]
        freq = {f"G{i}": 50 - i for i in range(50)}
        panels = {
            "hallmark_apoptosis": ["G0", "G1", "G2", "G3"],
            "k562_myeloid_lineage": ["G10", "G11", "G12", "G13"],
        }
        panel_sources = {
            "hallmark_apoptosis": {"collection": "hallmark", "source": "MSigDB synthetic"},
            "k562_myeloid_lineage": {"collection": "lineage", "source": "curated synthetic"},
        }

        df = run_depmap_enrichment(
            rl_action_freq=freq,
            background_genes=bg,
            panels=panels,
            panel_sources=panel_sources,
            chronos_df=self._mock_chronos_df(),
            top_k=10,
            n_null=30,
            expression_mean_per_gene={g: float(i) for i, g in enumerate(bg)},
        )

        assert {"panel_source", "n_panel_genes", "n_panel_in_background", "null_strategy", "status"}.issubset(df.columns)
        assert set(df["panel"].to_list()) >= {
            "depmap_k562_essentials",
            "hallmark_apoptosis",
            "k562_myeloid_lineage",
        }
        strategies = set(df["null_strategy"].to_list())
        assert "expression_matched" in strategies

    def test_empty_panel_is_reported_not_silently_dropped(self) -> None:
        from src.analysis.depmap_validation import run_depmap_enrichment

        bg = [f"G{i}" for i in range(20)]
        freq = {f"G{i}": 20 - i for i in range(20)}
        df = run_depmap_enrichment(
            rl_action_freq=freq,
            background_genes=bg,
            panels={"empty_overlap_panel": ["NOT_IN_BACKGROUND"]},
            chronos_df=self._mock_chronos_df(),
            top_k=5,
            n_null=10,
        )

        row = df.filter(df["panel"] == "empty_overlap_panel").to_dicts()[0]
        assert row["status"] == "no_background_overlap"
        assert row["p_value"] is None


# =============================================================================
# trajectory: load_rollouts schema validation
# =============================================================================


class TestLoadRollouts:
    def _make_rollout_parquet(self, tmp_path: Path) -> Path:
        import polars as pl

        path = tmp_path / "rollouts.parquet"
        n = 20
        rng = np.random.default_rng(0)
        df = pl.DataFrame({
            "episode_id": [0] * 10 + [1] * 10,
            "step":       list(range(10)) + list(range(10)),
            "action":     rng.integers(0, 5, n).tolist(),
            "gene_symbol": [f"G{i}" for i in rng.integers(0, 5, n)],
            "z_norm":     rng.uniform(0, 5, n).astype(np.float32).tolist(),
            "reward":     rng.normal(0, 1, n).astype(np.float32).tolist(),
            "terminated": ([False] * 9 + [True]) * 2,
            "success":    ([False] * 9 + [True]) + ([False] * 10),
            "z_vector":   [rng.normal(size=32).astype(np.float32).tolist() for _ in range(n)],
        })
        df.write_parquet(str(path))
        return path

    def test_loads_valid_file(self, tmp_path: Any) -> None:
        from src.analysis.trajectory import load_rollouts

        path = self._make_rollout_parquet(Path(tmp_path))
        df = load_rollouts(path)
        assert len(df) == 20
        assert "z_vector" in df.columns

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        from src.analysis.trajectory import load_rollouts

        with pytest.raises(FileNotFoundError):
            load_rollouts(tmp_path / "missing.parquet")

    def test_missing_columns_raises(self, tmp_path: Any) -> None:
        import polars as pl

        from src.analysis.trajectory import load_rollouts

        bad = tmp_path / "bad.parquet"
        pl.DataFrame({"episode_id": [0], "step": [0]}).write_parquet(str(bad))
        with pytest.raises(ValueError, match="Contract 4"):
            load_rollouts(bad)


# =============================================================================
# trajectory: project_rollouts_to_umap
# =============================================================================


class TestProjectRollouts:
    def test_adds_umap_columns(self) -> None:
        import polars as pl

        try:
            import umap as umap_mod
        except ImportError:
            pytest.skip("umap-learn not installed")

        from src.analysis.trajectory import project_rollouts_to_umap

        rng = np.random.default_rng(0)
        n = 15
        Z_bg = rng.normal(size=(50, 8)).astype(np.float32)
        reducer = umap_mod.UMAP(n_neighbors=5, random_state=0, n_jobs=1)
        reducer.fit(Z_bg)

        z_vecs = rng.normal(size=(n, 8)).astype(np.float32).tolist()
        df = pl.DataFrame({
            "episode_id": [0] * n,
            "step": list(range(n)),
            "z_vector": list(z_vecs),
        })
        result = project_rollouts_to_umap(df, reducer)
        assert "umap_x" in result.columns
        assert "umap_y" in result.columns
        assert len(result) == n
