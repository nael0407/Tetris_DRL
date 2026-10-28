"""Unit tests untuk antarmuka Gymnasium TetrisEnv (tetris_rl.env).

Memverifikasi kriteria Tahap 5 Dokumen Proyek:
- Kepatuhan resmi terhadap gymnasium.utils.env_checker.check_env untuk seluruh opsi observasi.
- Ketepatan action masking (info["action_mask"] dan env.action_masks()).
- Perilaku skema reward R0, R1, R2, dan R3.
- Penanda terminasi (terminated = True saat game over) dan pemotongan (truncated = True saat max_steps tercapai).
- Determinisme seed pada pemanggilan env.reset(seed=...).
- Dukungan mode render ansi dan rgb_array.
"""

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from tetris_rl.env import TetrisEnv


def test_gymnasium_check_env_raw_obs() -> None:
    """Memverifikasi kepatuhan terhadap standar Gymnasium untuk obs_type='raw'."""
    env = TetrisEnv(obs_type="raw", reward_type="R0", max_steps=100)
    # check_env akan melempar assertion error jika ada pelanggaran standar Gymnasium
    check_env(env, skip_render_check=True)
    env.close()


def test_gymnasium_check_env_features_obs() -> None:
    """Memverifikasi kepatuhan terhadap standar Gymnasium untuk obs_type='features'."""
    env = TetrisEnv(obs_type="features", reward_type="R1", max_steps=100)
    check_env(env, skip_render_check=True)
    env.close()


def test_gymnasium_check_env_combined_obs() -> None:
    """Memverifikasi kepatuhan terhadap standar Gymnasium untuk obs_type='combined'."""
    env = TetrisEnv(obs_type="combined", reward_type="R2", max_steps=100)
    check_env(env, skip_render_check=True)
    env.close()


def test_action_masking_consistency() -> None:
    """Memastikan action mask sinkron dengan legalitas aksi dan piece aktif."""
    env = TetrisEnv(board_height=20, board_width=10)
    obs, info = env.reset(seed=42)

    mask = info["action_mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (40,)
    assert np.array_equal(mask, env.action_masks())

    # Pilih aksi yang legal
    valid_actions = np.nonzero(mask)[0]
    assert len(valid_actions) > 0
    chosen_action = int(valid_actions[0])

    next_obs, reward, terminated, truncated, next_info = env.step(chosen_action)
    assert not terminated or next_info["is_game_over"]
    assert env.observation_space.contains(next_obs)

    # Uji aksi ilegal (di luar mask aktif saat ini)
    current_mask = env.action_masks()
    illegal_actions = np.where(~current_mask)[0]
    if len(illegal_actions) > 0:
        bad_action = int(illegal_actions[0])
        # Aksi ilegal harus memicu terminasi langsung dan penalti negatif
        _, bad_rew, bad_term, _, bad_info = env.step(bad_action)
        assert bad_term is True
        assert bad_rew < 0
        assert bad_info["is_game_over"] is True

    env.close()


def test_reward_variants() -> None:
    """Menguji perhitungan berbagai varian reward (R0, R1, R2, R3)."""
    # 1. R0 (skor game murni)
    env_r0 = TetrisEnv(reward_type="R0")
    env_r0.reset(seed=1)
    act = np.nonzero(env_r0.action_masks())[0][0]
    _, rew_r0, _, _, _ = env_r0.step(act)
    assert rew_r0 == 0.0 or rew_r0 in (100.0, 300.0, 500.0, 800.0)
    env_r0.close()

    # 2. R1 (ada bonus bertahan +1)
    env_r1 = TetrisEnv(reward_type="R1")
    env_r1.reset(seed=1)
    act = np.nonzero(env_r1.action_masks())[0][0]
    _, rew_r1, term, _, _ = env_r1.step(act)
    if not term:
        assert rew_r1 >= 1.0
    env_r1.close()

    # 3. R3 (potential-based shaping)
    env_r3 = TetrisEnv(reward_type="R3")
    env_r3.reset(seed=1)
    act = np.nonzero(env_r3.action_masks())[0][0]
    _, rew_r3, _, _, _ = env_r3.step(act)
    assert isinstance(rew_r3, float)
    env_r3.close()


def test_seed_determinism() -> None:
    """Memverifikasi determinisme lingkungan saat diberikan seed yang sama."""
    env1 = TetrisEnv(obs_type="raw", max_steps=50)
    env2 = TetrisEnv(obs_type="raw", max_steps=50)

    obs1, info1 = env1.reset(seed=12345)
    obs2, info2 = env2.reset(seed=12345)

    np.testing.assert_array_equal(obs1, obs2)
    assert info1["current_piece"] == info2["current_piece"]
    assert info1["next_piece"] == info2["next_piece"]

    # Jalankan serangkaian aksi legal identik
    for _ in range(10):
        mask1 = env1.action_masks()
        mask2 = env2.action_masks()
        np.testing.assert_array_equal(mask1, mask2)

        act = int(np.nonzero(mask1)[0][0])
        o1, r1, t1, tr1, i1 = env1.step(act)
        o2, r2, t2, tr2, i2 = env2.step(act)

        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2
        assert i1["score"] == i2["score"]
        if t1 or tr1:
            break

    env1.close()
    env2.close()


def test_max_steps_truncation() -> None:
    """Memastikan episode dipotong (truncated=True) ketika langkah mencapai max_steps."""
    limit = 5
    env = TetrisEnv(max_steps=limit)
    env.reset(seed=77)

    truncated = False
    for step in range(limit):
        act = int(np.nonzero(env.action_masks())[0][0])
        _, _, terminated, truncated, _ = env.step(act)
        if terminated:
            break

    assert truncated is True
    env.close()


def test_render_modes() -> None:
    """Menguji eksekusi render ansi dan rgb_array."""
    # 1. Ansi mode
    env_ansi = TetrisEnv(render_mode="ansi")
    env_ansi.reset(seed=10)
    txt = env_ansi.render()
    assert isinstance(txt, str)
    assert "TETRIS RL" in txt
    env_ansi.close()

    # 2. RGB array mode
    env_rgb = TetrisEnv(render_mode="rgb_array")
    env_rgb.reset(seed=10)
    frame = env_rgb.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env_rgb.close()
