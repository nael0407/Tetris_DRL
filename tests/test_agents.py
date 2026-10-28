"""Unit tests untuk agen-agen baseline (tetris_rl.agents) dan protokol evaluasi.

Memverifikasi:
- RandomAgent selalu memilih aksi yang valid sesuai action_mask.
- HeuristicAgent memilih aksi legal berbasis bobot linear afterstate.
- CEMAgent menjalankan loop optimasi populasi dan menghitung total piece interaksi.
- evaluate_agent mengeksekusi multi-seed evaluasi dan mengembalikan BenchmarkSummary valid.
"""

import numpy as np
import pytest

from tetris_rl.agents.cem import CEMAgent
from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.eval.evaluate import evaluate_agent


def test_random_agent() -> None:
    """Memastikan RandomAgent selalu memilih aksi legal dari mask."""
    agent = RandomAgent(seed=42)
    env = TetrisEnv(board_height=10, board_width=6)
    obs, info = env.reset(seed=10)

    for _ in range(20):
        mask = info["action_mask"]
        act = agent.act(obs=obs, info=info)
        assert mask[act], "RandomAgent memilih aksi ilegal!"

        obs, reward, terminated, truncated, info = env.step(act)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def test_heuristic_agent() -> None:
    """Memastikan HeuristicAgent mengevaluasi afterstate dan mengambil keputusan yang sah."""
    agent = HeuristicAgent()
    env = TetrisEnv(board_height=12, board_width=8)
    obs, info = env.reset(seed=99)

    for _ in range(15):
        mask = info["action_mask"]
        act = agent.act(env, obs=obs, info=info)
        assert mask[act]

        obs, reward, terminated, truncated, info = env.step(act)
        if terminated or truncated:
            break

    env.close()


def test_cem_agent_optimization_loop() -> None:
    """Memastikan CEMAgent mengoptimasi bobot populasi dan menghitung total piece interaksi."""
    env = TetrisEnv(board_height=10, board_width=6, max_steps=50)

    cem = CEMAgent(
        num_features=4,
        population_size=6,
        elite_ratio=0.33,
        initial_std=0.2,
        seed=123,
    )

    res = cem.train(
        env=env,
        max_iterations=2,
        episodes_per_candidate=1,
    )

    assert res.total_iterations == 2
    assert res.total_interaction_pieces > 0
    assert len(res.best_weights) == 4
    # Bobot ternormalisasi (norm L2 mendekati 1)
    np.testing.assert_almost_equal(np.linalg.norm(res.best_weights), 1.0, decimal=4)

    env.close()


def test_evaluate_agent_protocol() -> None:
    """Memastikan evaluate_agent menghasilkan BenchmarkSummary lengkap."""
    env = TetrisEnv(board_height=10, board_width=6, max_steps=30)
    agent = RandomAgent(seed=777)

    summary = evaluate_agent(
        agent=agent,
        env=env,
        seeds=[1, 2],
        episodes_per_seed=2,
        max_steps_per_episode=30,
        agent_name="Test-Random",
    )

    assert summary.agent_name == "Test-Random"
    assert summary.num_episodes == 4
    assert len(summary.episodes) == 4
    assert summary.score_stats.mean >= 0
    assert summary.lines_stats.mean >= 0
    assert len(summary.survival_timeline) == len(summary.survival_probabilities)

    env.close()
