"""Unit tests untuk modul analisis interpretabilitas (tetris_rl.analysis).

Memverifikasi:
- Ekstraksi fitur ground-truth (holes, max_height, bumpiness, wells).
- Ekstraksi aktivasi hidden layer model DQN berdimensi valid.
- Linear probing mencapai estimasi R^2 dan Pearson correlation yang valid.
- Intervensi kausal (suntik lubang dan tambah tinggi) memodifikasi papan dengan benar.
"""

import numpy as np
import pytest
import torch

from tetris_rl.agents.td_afterstate import DQNAfterstateAgent
from tetris_rl.analysis.intervention import (
    evaluate_board_value,
    increase_column_height,
    inject_hole,
    run_intervention_analysis,
)
from tetris_rl.analysis.probing import (
    compute_wells,
    extract_ground_truth_features,
    extract_hidden_activations,
    fit_linear_probe,
    run_linear_probing,
)
import tetris_rl.board as board_module
from tetris_rl.env import TetrisEnv


def test_ground_truth_features_and_wells() -> None:
    """Menguji perhitungan ground-truth features dan identifikasi sumur."""
    board = board_module.create_board(height=10, width=6)
    # Kolom 0-4 terisi di baris 9, kolom 5 kosong (membentuk sumur 1-lebar di kolom 5)
    board[9, :5] = 1
    # Buat lubang di kolom 2 di baris 8
    board[7, 2] = 1

    gt = extract_ground_truth_features(board)
    assert "holes" in gt
    assert "max_height" in gt
    assert "wells" in gt
    assert gt["holes"] >= 1.0
    assert gt["wells"] >= 1.0


def test_extract_hidden_activations() -> None:
    """Memastikan ekstraksi aktivasi hidden layer mengembalikan matriks (N, hidden_dim)."""
    agent = DQNAfterstateAgent(
        board_height=10,
        board_width=6,
        representation="raw",
        hidden_dims=(32, 16),
        seed=42,
    )

    boards = np.zeros((5, 10, 6), dtype=np.uint8)
    boards[0, 9, :] = 1
    boards[1, 8, :] = 1

    acts_layer0 = extract_hidden_activations(agent, boards, layer_idx=0)
    assert isinstance(acts_layer0, np.ndarray)
    assert acts_layer0.shape == (5, 32)
    assert not np.all(acts_layer0 == 0)


def test_fit_linear_probe_synthetic() -> None:
    """Menguji estimasi solusi least-squares linear probe pada data sintetis linear sempurna."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10)).astype(np.float32)
    true_w = rng.normal(size=(10,)).astype(np.float32)
    y = X @ true_w + 2.5  # Linear dengan bias

    res = fit_linear_probe(X[:80], y[:80], X[80:], y[80:], l2_reg=0.001)
    assert res.r2_score > 0.95
    assert res.pearson_r > 0.95
    assert res.mse < 0.1


def test_causal_interventions_mutation() -> None:
    """Memastikan intervensi lubang dan ketinggian memodifikasi papan secara valid."""
    board = board_module.create_board(height=10, width=6)
    board[8:10, 2] = 1  # 2 blok bertumpuk di kolom 2

    # Suntik lubang
    b_hole, ok_h = inject_hole(board)
    assert ok_h
    assert np.sum(b_hole) == np.sum(board) - 1

    # Tambah tinggi
    b_height, ok_ht = increase_column_height(board, num_blocks=2)
    assert ok_ht
    assert np.sum(b_height) == np.sum(board) + 2


def test_end_to_end_analysis_pipeline() -> None:
    """Menguji eksekusi end-to-end linear probing dan analisis intervensi."""
    env = TetrisEnv(board_height=10, board_width=6)
    agent = DQNAfterstateAgent(
        board_height=10,
        board_width=6,
        representation="raw",
        hidden_dims=(32, 16),
        seed=101,
    )

    # 1. Probing
    probing_res = run_linear_probing(agent, env, num_samples=100, test_ratio=0.3, layer_idx=0)
    assert "holes" in probing_res
    assert "max_height" in probing_res
    assert 0.0 <= probing_res["holes"].r2_score <= 1.0

    # 2. Intervensi
    boards = [env.reset(seed=i)[0][:60].reshape(10, 6).astype(np.uint8) for i in range(10)]
    interv_res = run_intervention_analysis(agent, boards)
    assert "hole_injection" in interv_res
    assert "height_increase" in interv_res
    assert isinstance(interv_res["hole_injection"].mean_delta_value, float)

    env.close()

