"""CellPath — RL agent for in-silico cell-state steering over a CRISPRa surrogate environment.

Top-level package. See ARCHITECTURE.md for the system diagram and CLAUDE.md for sacred rules.

Subpackages
-----------
- ``data``     : Norman 2019 download + preprocessing + OT pseudo-pairing (Agent A).
- ``models``   : scVI VAE wrapper + dynamics MLP + gene embeddings (Agents A and B).
- ``rl``       : Gymnasium environment + MaskablePPO trainer + reward (Agent B).
- ``analysis`` : Latent-space metrics, trajectory rendering, DepMap enrichment,
                 single-source-of-truth ``metrics.py``.
- ``utils``    : Device detection, seeding, checkpointing, logging (shared).

End-to-end orchestration lives in :mod:`src.pipeline`.
"""

__version__ = "0.1.0"
