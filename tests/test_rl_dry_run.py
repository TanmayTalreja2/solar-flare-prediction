import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rl_dry_run import base_param, q_margin, skill, train_double_q


def test_agent_alerts_only_in_high_flare_state():
    rng = np.random.default_rng(1)
    n = 3000
    states = rng.integers(0, 2, n)
    labels = np.where(states == 0, rng.random(n) < 0.9, rng.random(n) < 0.01).astype(int)
    bounds = [(i, min(i + 50, n)) for i in range(0, n, 50)]

    qa, qb = train_double_q(states, labels, bounds, 2, (5.0, 1.0, -1.0, -8.0), epochs=8, seed=0)
    margin = q_margin(qa, qb)

    assert margin[0] > 0
    assert margin[1] < 0


def test_skill_scores():
    s = skill([1, 1, 0, 0], [1, 0, 1, 0])
    assert s["recall"] == 0.5 and s["far"] == 0.5 and s["tss"] == 0.0
    assert skill([1, 0], [1, 0])["tss"] == 1.0


def test_base_param():
    assert base_param("LOG_TOTUSJH") == "TOTUSJH"
    assert base_param("ABSNJZH_ROLLSTD_12h") == "ABSNJZH"
    assert base_param("SAVNCPP_RELCHANGE_3h") == "SAVNCPP"
    assert base_param("R_VALUE") == "R_VALUE"