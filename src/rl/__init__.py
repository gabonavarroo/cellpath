"""RL environment + MaskablePPO trainer + reward shaping.

Owner: Agent B. See ARCHITECTURE.md Concepts 4 + 5 and AGENTS.md §2.

Modules
-------
- :mod:`src.rl.environment` : ``CellReprogrammingEnv`` (gymnasium env).
- :mod:`src.rl.train_ppo`   : ``MaskablePPO`` trainer (sb3-contrib).
- :mod:`src.rl.reward`      : Distance + sparsity + uncertainty terms.
"""
