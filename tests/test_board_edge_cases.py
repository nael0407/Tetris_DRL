"""Pengujian kasus ekstrem (edge cases) untuk logika papan Tetris.

Memverifikasi skenario kritis sesuai kriteria Tahap 3:
- Pembersihan multi-baris tak bersambung (discontinuous line clears) dan penurunan baris berongga.
- Preservasi lubang (holes) dan struktur menggantung (overhangs) saat baris di bawahnya terhapus.
- Validasi seluruh 19 rotasi unik pada seluruh kolom legal dan penolakan batas dinding.
- Presisi batas Game Over pada row 0 vs row -1.
"""

import numpy as np
import pytest

from tetris_rl.board import (
    compute_landing_row,
    create_board,
    get_action_mask,
    get_column_tops,
    place_piece,
)
from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    NUM_PIECES,
    PIECE_NAME_TO_ID,
    PIECES,
    get_piece,
    get_piece_rotation,
)


def test_discontinuous_multi_line_clearing() -> None:
    """Menguji penghapusan baris terpisah (misal baris 16 dan baris 19 penuh, tapi 17 dan 18 tidak)."""
    board = create_board(height=20, width=10)

    # Buat baris 19 penuh
    board[19, :] = 1
    # Baris 18: sebagian terisi (kolom 0 dan 1)
    board[18, 0:2] = 1
    # Baris 17: kolom 4..7 terisi (sebagai penumpu piece di baris 16), kolom lain kosong
    board[17, 4:8] = 1
    # Baris 16: hampir penuh kecuali kolom 4..7 (bolong tepat 4 sel untuk diisi piece I)
    board[16, :] = 1
    board[16, 4:8] = 0
    # Baris 15: ada blok penanda di kolom 0
    board[15, 0] = 1

    # Tempatkan I horizontal (lebar 4) di kolom 4
    # Karena kolom 4..7 di baris 17 terisi blok, piece tertahan dan mendarat tepat di baris 16!
    # Pendaratan ini melengkapi baris 16 menjadi penuh 100%.
    id_i = PIECE_NAME_TO_ID["I"]
    res = place_piece(board, piece_id=id_i, rotation=0, col=4)

    assert not res.is_game_over
    assert res.landing_row == 16
    # Baris 16 dan baris 19 sama-sama penuh -> tepat 2 baris terhapus!
    assert res.lines_cleared == 2

    # Analisis posisi setelah penurunan:
    # 1. Baris 18 lama (kolom 0..1) turun 1 tingkat ke baris 19
    assert np.all(res.board[19, 0:2] == 1)
    assert np.sum(res.board[19, :]) == 2

    # 2. Baris 17 lama (kolom 4..7) turun 1 tingkat ke baris 18
    assert np.all(res.board[18, 4:8] == 1)
    assert np.sum(res.board[18, :]) == 4

    # 3. Baris 15 lama (blok di kolom 0) turun 2 tingkat ke baris 17
    assert res.board[17, 0] == 1
    assert np.sum(res.board[17, :]) == 1

    # Seluruh baris dari 0 sampai 16 sekarang kosong (nol)
    assert np.all(res.board[0:17, :] == 0)


def test_overhang_and_hole_preservation() -> None:
    """Memastikan lubang di bawah blok tetap terjaga saat baris di bawahnya dihapus."""
    board = create_board(height=20, width=10)

    # Baris 19 penuh kecuali kolom 0
    board[19, 1:10] = 1
    # Baris 18 kosong (ini akan jadi lubang/overhang)
    # Baris 17 memiliki blok "atap" di kolom 3
    board[17, 3] = 1
    # Baris 16 memiliki blok penanda di kolom 3
    board[16, 3] = 1

    # Jatuhkan piece I vertikal di kolom 0 -> memenuhi baris 19
    id_i = PIECE_NAME_TO_ID["I"]
    res = place_piece(board, piece_id=id_i, rotation=1, col=0)

    assert not res.is_game_over
    assert res.lines_cleared == 1

    # Baris 19 lama terhapus. Seluruh blok di atas baris 19 turun tepat 1 baris
    # Blok di baris 17 kolom 3 sekarang berada di baris 18 kolom 3
    assert res.board[18, 3] == 1
    # Blok di baris 16 kolom 3 sekarang berada di baris 17 kolom 3
    assert res.board[17, 3] == 1
    # Kolom 3 baris 19 harus KOSONG (karena sebelumnya baris 18 kolom 3 kosong!)
    assert res.board[19, 3] == 0, "Lubang di bawah blok tertutup secara keliru!"


def test_all_19_rotations_all_legal_columns() -> None:
    """Menguji penempatan seluruh 19 rotasi unik pada seluruh kolom legal."""
    board = create_board(height=20, width=10)

    total_placements_tested = 0
    for p in PIECES:
        for rot in p.rotations:
            p_width = rot.width
            max_legal_col = BOARD_WIDTH_DEFAULT - p_width

            for col in range(max_legal_col + 1):
                res = place_piece(board, piece_id=p.id, rotation=rot.rotation_idx, col=col)
                assert not res.is_game_over
                assert res.landing_row == 20 - rot.height
                assert np.sum(res.board) == 4
                total_placements_tested += 1

            # Menguji penolakan di luar batas kanan (col = max_legal_col + 1)
            with pytest.raises(ValueError):
                place_piece(board, piece_id=p.id, rotation=rot.rotation_idx, col=max_legal_col + 1)

            # Menguji penolakan di luar batas kiri (col = -1)
            with pytest.raises(ValueError):
                place_piece(board, piece_id=p.id, rotation=rot.rotation_idx, col=-1)

    # Total penempatan legal untuk seluruh piece pada papan 10x20
    # I: (7 + 10) = 17
    # O: 9
    # T: (8 + 9 + 8 + 9) = 34
    # S: (8 + 9) = 17
    # Z: (8 + 9) = 17
    # J: (8 + 9 + 8 + 9) = 34
    # L: (8 + 9 + 8 + 9) = 34
    # Total = 17 + 9 + 34 + 17 + 17 + 34 + 34 = 162
    assert total_placements_tested == 162


def test_game_over_boundary_precision() -> None:
    """Memverifikasi batas presisi Game Over: landing_row == 0 sah, landing_row < 0 Game Over."""
    board = create_board(height=20, width=10)
    id_i = PIECE_NAME_TO_ID["I"]

    # Isi kolom 0 mulai dari baris 4 sampai 19 (tinggi 16 blok)
    board[4:20, 0] = 1

    # I vertikal memiliki tinggi 4.
    # Jika diletakkan di kolom 0, landing_row = top[0] - 1 - 3 = 4 - 4 = 0.
    # Piece akan menempati baris 0, 1, 2, 3. Tepat muat di batas atas papan!
    res_fit = place_piece(board, piece_id=id_i, rotation=1, col=0)
    assert not res_fit.is_game_over, "Piece yang tepat muat di baris 0 dianggap Game Over!"
    assert res_fit.landing_row == 0
    assert np.all(res_fit.board[0:4, 0] == 1)

    # Sekarang, jika kolom 0 sudah terisi mulai dari baris 3 (tinggi 17 blok):
    board[3:20, 0] = 1
    # I vertikal diletakkan di kolom 0: landing_row = 3 - 4 = -1 (menembus baris atas) -> Game Over!
    res_overflow = place_piece(board, piece_id=id_i, rotation=1, col=0)
    assert res_overflow.is_game_over, "Piece yang menembus baris -1 tidak memicu Game Over!"
    assert res_overflow.landing_row == -1
    # Matriks papan tidak boleh rusak
    np.testing.assert_array_equal(res_overflow.board, board)


def test_completely_full_board_game_over() -> None:
    """Papan yang terisi penuh harus langsung memicu Game Over untuk piece apapun."""
    board = np.ones((BOARD_HEIGHT_DEFAULT, BOARD_WIDTH_DEFAULT), dtype=np.uint8)
    for p_id in range(NUM_PIECES):
        res = place_piece(board, piece_id=p_id, rotation=0, col=0)
        assert res.is_game_over
        assert res.landing_row < 0
