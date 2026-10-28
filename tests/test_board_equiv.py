"""Pengujian kesetaraan (behavioral equivalence test) antara tetris_rl.board (NumPy murni)
dan tetris_rl.board_fast (Numba terakselerasi).

Sesuai kriteria selesai Tahap 2 Dokumen Proyek:
- Versi numba WAJIB setara dengan versi referensi pada ribuan konfigurasi papan acak.
- Seluruh keluaran (landing_row, resulting_board, lines_cleared, is_game_over, valid_placements)
  harus 100% identik tanpa deviasi sedikit pun.
"""

import numpy as np
import pytest

import tetris_rl.board as ref_board
import tetris_rl.board_fast as fast_board
from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    NUM_PIECES,
    get_piece,
)


def _generate_random_board(
    rng: np.random.Generator,
    height: int = 20,
    width: int = 10,
    density: float = 0.3,
    max_height: int = 16,
    force_almost_full_rows: bool = False,
) -> np.ndarray:
    """Helper untuk membangkitkan konfigurasi papan acak realistis."""
    board = np.zeros((height, width), dtype=np.uint8)

    if force_almost_full_rows:
        # Papan dengan baris terbawah hampir penuh (hanya bolong 1-4 sel)
        # dan area di atasnya kosong agar piece mendarat langsung melengkapi baris penuh
        num_almost_full = int(rng.integers(1, 5))
        for offset in range(num_almost_full):
            r = height - 1 - offset
            board[r, :] = 1
            missing_cols = rng.choice(width, size=int(rng.integers(1, 4)), replace=False)
            board[r, missing_cols] = 0
    else:
        # Isi blok acak di baris bawah (H - max_height sampai H - 1)
        for r in range(height - max_height, height):
            mask = rng.random(width) < density
            board[r, mask] = 1

    # Pastikan tidak ada baris yang sudah penuh 100% sebelum piece dijatuhkan
    for r in range(height):
        if np.all(board[r] == 1):
            board[r, rng.integers(0, width)] = 0

    return board


def test_column_tops_equivalence() -> None:
    """Menguji kesetaraan get_column_tops pada 1.000 papan acak."""
    rng = np.random.default_rng(seed=42)
    for _ in range(1000):
        density = rng.uniform(0.1, 0.8)
        max_h = rng.integers(1, 19)
        board = _generate_random_board(rng, density=density, max_height=max_h)

        ref_tops = ref_board.get_column_tops(board)
        fast_tops = fast_board.get_column_tops(board)

        np.testing.assert_array_equal(
            ref_tops, fast_tops, err_msg="get_column_tops tidak identik antara NumPy dan Numba"
        )


def test_compute_landing_row_equivalence() -> None:
    """Menguji kesetaraan compute_landing_row pada 1.000 skenario acak."""
    rng = np.random.default_rng(seed=123)
    for _ in range(1000):
        board = _generate_random_board(rng, density=rng.uniform(0.1, 0.7), max_height=rng.integers(1, 18))
        piece_id = int(rng.integers(0, NUM_PIECES))
        p = get_piece(piece_id)
        rot_idx = int(rng.integers(0, p.num_rotations))
        p_rot = p.rotations[rot_idx]
        col = int(rng.integers(0, BOARD_WIDTH_DEFAULT - p_rot.width + 1))

        ref_landing = ref_board.compute_landing_row(board, p_rot.coords, col)
        fast_landing = fast_board.compute_landing_row(board, p_rot.coords, col)

        assert ref_landing == fast_landing, (
            f"Landing row berbeda untuk piece {p.name} rot {rot_idx} col {col}: "
            f"ref={ref_landing}, fast={fast_landing}"
        )


def test_place_piece_equivalence_thousands_boards() -> None:
    """Menguji kesetaraan place_piece pada 2.500 variasi papan acak (termasuk kondisi Game Over dan Line Clears)."""
    rng = np.random.default_rng(seed=999)
    total_clears_tested = 0
    total_game_overs_tested = 0

    for iteration in range(2500):
        # Variasikan kepadatan dan ketinggian tumpukan dari rendah hingga sangat tinggi
        density = rng.uniform(0.1, 0.95)
        # Seringkali uji papan yang tinggi (max_height hingga 20) untuk menguji game over
        max_h = rng.integers(1, 21)
        force_lines = (iteration % 2 == 0)
        board = _generate_random_board(
            rng, density=density, max_height=max_h, force_almost_full_rows=force_lines
        )

        piece_id = int(rng.integers(0, NUM_PIECES))
        p = get_piece(piece_id)
        rot_idx = int(rng.integers(0, p.num_rotations))
        p_rot = p.rotations[rot_idx]
        col = int(rng.integers(0, BOARD_WIDTH_DEFAULT - p_rot.width + 1))

        if iteration % 4 == 0:
            # Rekayasa baris terbawah agar bolong tepat selebar piece di kolom 'col'
            # Ini memastikan ratusan kasus line clear multi-baris teruji secara deterministik
            board[19, :] = 1
            board[19, col : col + p_rot.width] = 0
            if iteration % 8 == 0 and p_rot.height > 1:
                board[18, :] = 1
                board[18, col : col + p_rot.width] = 0

        ref_res = ref_board.place_piece(board, piece_id, rot_idx, col)
        fast_res = fast_board.place_piece(board, piece_id, rot_idx, col)

        # 1. Cek status Game Over
        assert ref_res.is_game_over == fast_res.is_game_over, (
            f"Iter {iteration}: is_game_over beda (ref={ref_res.is_game_over}, fast={fast_res.is_game_over})"
        )
        if ref_res.is_game_over:
            total_game_overs_tested += 1

        # 2. Cek baris terhapus
        assert ref_res.lines_cleared == fast_res.lines_cleared, (
            f"Iter {iteration}: lines_cleared beda (ref={ref_res.lines_cleared}, fast={fast_res.lines_cleared})"
        )
        if ref_res.lines_cleared > 0:
            total_clears_tested += ref_res.lines_cleared

        # 3. Cek posisi pendaratan
        assert ref_res.landing_row == fast_res.landing_row, (
            f"Iter {iteration}: landing_row beda (ref={ref_res.landing_row}, fast={fast_res.landing_row})"
        )

        # 4. Cek kesamaan mutlak matriks papan hasil
        np.testing.assert_array_equal(
            ref_res.board,
            fast_res.board,
            err_msg=f"Iter {iteration}: Matriks papan hasil berbeda antara NumPy dan Numba",
        )

    # Pastikan pengujian benar-benar menemui situasi non-trivial
    assert total_clears_tested > 50, f"Line clear terlalu sedikit teruji: {total_clears_tested}"
    assert total_game_overs_tested > 50, f"Game over terlalu sedikit teruji: {total_game_overs_tested}"


def test_valid_placements_equivalence() -> None:
    """Menguji kesetaraan kalkulasi seluruh afterstate (get_valid_placements) pada 200 papan acak."""
    rng = np.random.default_rng(seed=777)
    for _ in range(200):
        board = _generate_random_board(rng, density=rng.uniform(0.1, 0.8), max_height=rng.integers(1, 18))
        piece_id = int(rng.integers(0, NUM_PIECES))

        ref_placements = ref_board.get_valid_placements(board, piece_id)
        fast_placements = fast_board.get_valid_placements(board, piece_id)

        assert len(ref_placements) == len(fast_placements), (
            f"Jumlah valid placements berbeda: ref={len(ref_placements)}, fast={len(fast_placements)}"
        )

        for i in range(len(ref_placements)):
            r = ref_placements[i]
            f = fast_placements[i]

            assert r.action == f.action
            assert r.rotation == f.rotation
            assert r.column == f.column
            assert r.lines_cleared == f.lines_cleared
            assert r.is_game_over == f.is_game_over
            assert r.landing_row == f.landing_row
            np.testing.assert_array_equal(r.next_board, f.next_board)
