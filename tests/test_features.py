"""Unit tests untuk modul fitur heuristik (tetris_rl.features).

Memverifikasi:
- Perhitungan profil ketinggian kolom, max height, aggregate height.
- Deteksi lubang (holes) di bawah blok terisi.
- Perhitungan kontur kebergelombangan (bumpiness).
- Konsistensi struktur BoardFeatures dan extract_feature_vector.
"""

import numpy as np
import pytest

from tetris_rl.board import create_board
from tetris_rl.features import (
    BoardFeatures,
    compute_aggregate_height,
    compute_bumpiness,
    compute_column_heights,
    compute_holes,
    compute_max_height,
    extract_feature_vector,
    get_board_features,
)


def test_empty_board_features() -> None:
    """Papan kosong harus memiliki semua metrik bernilai nol."""
    board = create_board(height=20, width=10)
    col_h = compute_column_heights(board)
    assert np.all(col_h == 0)
    assert compute_max_height(board) == 0
    assert compute_aggregate_height(board) == 0
    assert compute_holes(board) == 0
    assert compute_bumpiness(board) == 0

    feat_vec = extract_feature_vector(board, lines_cleared=0)
    assert feat_vec.shape == (15,)  # 10 col_heights + 5 scalar features
    assert np.all(feat_vec == 0.0)


def test_stair_and_bumpiness_features() -> None:
    """Papan bertangga: kolom berundak naik teratur."""
    board = create_board(height=10, width=5)
    # Kolom 0: tinggi 1 (baris 9 terisi)
    board[9, 0] = 1
    # Kolom 1: tinggi 2 (baris 8..9 terisi)
    board[8:10, 1] = 1
    # Kolom 2: tinggi 3 (baris 7..9 terisi)
    board[7:10, 2] = 1
    # Kolom 3: tinggi 4 (baris 6..9 terisi)
    board[6:10, 3] = 1
    # Kolom 4: tinggi 2 (baris 8..9 terisi)
    board[8:10, 4] = 1

    # col_heights: [1, 2, 3, 4, 2]
    np.testing.assert_array_equal(compute_column_heights(board), [1, 2, 3, 4, 2])
    assert compute_max_height(board) == 4
    assert compute_aggregate_height(board) == 1 + 2 + 3 + 4 + 2  # 12
    # Bumpiness: |1-2| + |2-3| + |3-4| + |4-2| = 1 + 1 + 1 + 2 = 5
    assert compute_bumpiness(board) == 5
    assert compute_holes(board) == 0


def test_holes_detection() -> None:
    """Memverifikasi deteksi lubang pada berbagai konfigurasi."""
    board = create_board(height=10, width=4)

    # Kolom 0: blok di baris 6 dan 8, baris 7 dan 9 kosong -> 2 lubang di baris 7 dan 9!
    board[6, 0] = 1
    board[8, 0] = 1

    # Kolom 1: blok di baris 5, baris 6..9 kosong -> 4 lubang (6, 7, 8, 9)!
    board[5, 1] = 1

    # Kolom 2: blok di baris 9 saja -> 0 lubang
    board[9, 2] = 1

    # Kolom 3: kosong -> 0 lubang

    # Total lubang: 2 + 4 + 0 + 0 = 6
    assert compute_holes(board) == 6

    bf = get_board_features(board, lines_cleared=0)
    assert bf.holes == 6
    assert bf.max_height == 5  # kolom 1 (10 - 5 = 5)

