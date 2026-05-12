"""Data pipeline: download + preprocess + OT pseudo-pairing.

Owner: Agent A. See AGENTS.md §1 and DATA.md for the full spec.

Modules
-------
- :mod:`src.data.download`            : pertpy primary + scperturb / GEO fallback for Norman 2019.
- :mod:`src.data.preprocess`          : scanpy pipeline; raw counts preserved in
                                        ``adata.layers["counts"]``.
- :mod:`src.data.perturbation_pairs`  : OT / random / mean-delta pairer; also exposes
                                        ``generate_mock_pairs`` for unblocking Agent B on Day 0.
"""
