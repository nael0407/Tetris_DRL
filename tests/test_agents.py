"""Unit tests untuk agen-agen baseline (tetris_rl.agents) dan protokol evaluasi.

Memverifikasi:
- RandomAgent selalu memilih aksi yang valid sesuai action_mask.
- HeuristicAgent memilih aksi legal berbasis bobot linear afterstate.
- CEMAgent menjalankan loop optimasi populasi dan menghitung total piece interaksi.
- evaluate_agent mengeksekusi multi-seed evaluasi dan mengembalikan BenchmarkSummary valid.
"""

import tempfile
import os
import numpy as np
import pytest
import torch

from tetris_rl.agents.cem import CEMAgent
from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.agents.ppo_masked import MaskedPPOAgent, make_masked_tetris_env
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.agents.td_afterstate import (
    DQNAfterstateAgent,
    LinearTDAfterstateAgent,
)
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


def test_linear_td_afterstate_agent() -> None:
    """Menguji LinearTDAfterstateAgent: ekstraksi fitur, evaluasi aksi, dan update TD(0)."""
    env = TetrisEnv(board_height=10, board_width=6)
    agent = LinearTDAfterstateAgent(alpha=0.01, seed=42)

    obs, info = env.reset(seed=12)
    placements = env.valid_placements()
    assert len(placements) > 0

    # Uji ekstraksi fitur
    phi = agent.extract_features(placements[0])
    assert len(phi) == 4
    val = agent.predict_value(phi)
    assert isinstance(val, float)

    # Uji pemilihan aksi deterministik
    act = agent.act(env=env, deterministic=True)
    mask = env.action_masks()
    assert mask[act], "LinearTDAfterstateAgent memilih aksi ilegal!"

    # Uji update TD(0)
    w_before = agent.weights.copy()
    td_err = agent.update(phi, reward=1.0, features_next=phi, done=False)
    assert isinstance(td_err, float)
    assert not np.array_equal(w_before, agent.weights), "Bobot harus terupdate setelah TD step!"

    # Uji save & load
    with tempfile.TemporaryDirectory() as tmpdir:
        w_path = os.path.join(tmpdir, "td_weights.npy")
        agent.save(w_path)
        assert os.path.exists(w_path)

        agent_loaded = LinearTDAfterstateAgent()
        agent_loaded.load(w_path)
        np.testing.assert_array_equal(agent.weights, agent_loaded.weights)

    env.close()


def test_dqn_afterstate_agent() -> None:
    """Menguji DQNAfterstateAgent: forward pass, action selection, replay buffer, gradien, checkpoint."""
    env = TetrisEnv(board_height=10, board_width=6)
    agent = DQNAfterstateAgent(
        board_height=10,
        board_width=6,
        representation="raw",
        hidden_dims=(32, 16),
        lr=1e-3,
        buffer_capacity=1000,
        batch_size=8,
        seed=42,
    )

    obs, info = env.reset(seed=55)
    act = agent.act(env=env, deterministic=True)
    mask = env.action_masks()
    assert mask[act], "DQNAfterstateAgent memilih aksi ilegal!"

    # Isi buffer mini
    placements = env.valid_placements()
    p0 = placements[0]
    for _ in range(15):
        agent.remember(
            board_after=p0.next_board,
            lines_cleared=p0.lines_cleared,
            reward=1.0,
            next_board_after=p0.next_board,
            next_lines_cleared=0,
            done=False,
        )

    assert len(agent.buffer) == 15
    loss = agent.update()
    assert loss is not None
    assert np.isfinite(loss)
    assert loss >= 0.0

    # Uji save & load checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dqn_test.pt")
        agent.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        agent2 = DQNAfterstateAgent(
            board_height=10,
            board_width=6,
            representation="raw",
            hidden_dims=(32, 16),
        )
        agent2.load(ckpt_path)
        # Verifikasi kesamaan bobot policy net
        p1 = list(agent.policy_net.parameters())[0].detach().cpu().numpy()
        p2 = list(agent2.policy_net.parameters())[0].detach().cpu().numpy()
        np.testing.assert_array_equal(p1, p2)

    env.close()


def test_masked_ppo_agent() -> None:
    """Menguji integrasi MaskablePPO dengan ActionMasker dan wrapper MaskedPPOAgent."""
    train_env = make_masked_tetris_env(board_height=10, board_width=6, obs_type="raw", seed=42)

    from sb3_contrib import MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        n_steps=32,
        batch_size=16,
        n_epochs=2,
        verbose=0,
        seed=42,
    )
    # Latih 32 langkah untuk memverifikasi loop integrasi
    model.learn(total_timesteps=32)

    agent = MaskedPPOAgent(model=model)

    test_env = TetrisEnv(board_height=10, board_width=6, obs_type="raw")
    obs, info = test_env.reset(seed=99)

    action = agent.act(env=test_env, obs=obs, info=info, deterministic=True)
    mask = test_env.action_masks()
    assert mask[action], "MaskedPPOAgent memilih aksi ilegal di luar action mask!"

    # Uji save & load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "ppo_test.zip")
        agent.save(model_path)
        assert os.path.exists(model_path)

        agent_loaded = MaskedPPOAgent(model_path=model_path)
        action2 = agent_loaded.act(env=test_env, obs=obs, info=info, deterministic=True)
        assert action2 == action

    train_env.close()
    test_env.close()
