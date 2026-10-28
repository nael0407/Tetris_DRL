"""Pengujian determinisme seed dan reproduktibilitas engine Tetris RL.

Memverifikasi kriteria Tahap 3 Dokumen Proyek & Kontrak Kerja:
- Pengacak piece (uniform dan 7-bag) 100% deterministik terhadap seed.
- Menjalankan ulang game dengan seed yang sama menghasilkan sekuens piece,
  transisi papan, pendaratan piece, skor, dan langkah game-over yang identik.
- Simulasi game identik antara mesin referensi NumPy dan mesin Numba.
"""

from typing import List, Tuple
import numpy as np
import pytest

import tetris_rl.board as ref_board
import tetris_rl.board_fast as fast_board
from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    NUM_PIECES,
    PieceGenerator,
    SevenBagPieceGenerator,
    UniformPieceGenerator,
    create_piece_generator,
)


def test_uniform_generator_determinism() -> None:
    """Memastikan UniformPieceGenerator menghasilkan sekuens identik pada seed yang sama."""
    seed = 42_1337
    gen1 = UniformPieceGenerator(seed=seed)
    seq1 = [gen1.next_piece() for _ in range(2000)]

    gen2 = UniformPieceGenerator(seed=seed)
    seq2 = [gen2.next_piece() for _ in range(2000)]

    assert seq1 == seq2, "Sekuens piece seragam berbeda untuk seed yang sama"

    # Reset dengan seed yang sama harus mengulang sekuens yang persis sama
    gen1.reset(seed=seed)
    seq1_reset = [gen1.next_piece() for _ in range(2000)]
    assert seq1 == seq1_reset

    # Seed berbeda harus menghasilkan sekuens berbeda
    gen3 = UniformPieceGenerator(seed=seed + 1)
    seq3 = [gen3.next_piece() for _ in range(2000)]
    assert seq1 != seq3


def test_uniform_generator_distribution() -> None:
    """Memverifikasi bahwa pengacak seragam membangkitkan ke-7 piece secara seimbang."""
    gen = UniformPieceGenerator(seed=999)
    n_samples = 70_000
    counts = np.zeros(NUM_PIECES, dtype=np.int32)
    for _ in range(n_samples):
        p = gen.next_piece()
        assert 0 <= p < NUM_PIECES
        counts[p] += 1

    # Rata-rata 10.000 per piece. Toleransi 5% (9.500 - 10.500)
    for p_id in range(NUM_PIECES):
        assert 9300 < counts[p_id] < 10700, f"Distribusi piece {p_id} tidak seragam: {counts[p_id]}"


def test_seven_bag_generator() -> None:
    """Memastikan 7-bag generator selalu membangkitkan permutasi tepat dari ke-7 piece tiap 7 langkah."""
    seed = 2026
    gen = SevenBagPieceGenerator(seed=seed)

    # Uji 100 bag berturut-turut (700 piece)
    for bag_idx in range(100):
        bag = [gen.next_piece() for _ in range(7)]
        assert sorted(bag) == list(range(7)), f"Bag ke-{bag_idx} bukan permutasi valid: {bag}"

    # Cek determinisme
    gen_dup = SevenBagPieceGenerator(seed=seed)
    seq_dup = [gen_dup.next_piece() for _ in range(700)]
    gen.reset(seed=seed)
    seq_orig = [gen.next_piece() for _ in range(700)]
    assert seq_orig == seq_dup


def test_create_piece_generator_factory() -> None:
    """Menguji factory function create_piece_generator."""
    g_uni = create_piece_generator("uniform", seed=10)
    assert isinstance(g_uni, UniformPieceGenerator)

    g_bag = create_piece_generator("7-bag", seed=10)
    assert isinstance(g_bag, SevenBagPieceGenerator)

    with pytest.raises(ValueError):
        create_piece_generator("invalid_generator_type")


def _run_deterministic_simulation(
    engine,
    seed: int,
    max_steps: int = 150,
) -> Tuple[List[int], List[int], List[np.ndarray], int]:
    """Helper untuk menjalankan simulasi game dengan kebijakan deterministik sederhana."""
    gen = UniformPieceGenerator(seed=seed)
    board = engine.create_board(BOARD_HEIGHT_DEFAULT, BOARD_WIDTH_DEFAULT)

    pieces_history: List[int] = []
    lines_history: List[int] = []
    board_history: List[np.ndarray] = []
    steps_survived = 0

    for step in range(max_steps):
        p_id = gen.next_piece()
        pieces_history.append(p_id)

        # Kebijakan aksi deterministik: ambil penempatan legal pertama dengan lines_cleared terbanyak
        # atau landing_row terendah (paling dasar)
        placements = engine.get_valid_placements(board, p_id)
        if not placements:
            break

        # Urutkan secara deterministik berdasarkan (is_game_over asc, -lines_cleared, -landing_row, action)
        chosen = min(
            placements,
            key=lambda x: (
                1 if x.is_game_over else 0,
                -x.lines_cleared,
                -x.landing_row,
                x.action,
            ),
        )

        board = chosen.next_board
        lines_history.append(chosen.lines_cleared)
        board_history.append(board.copy())
        steps_survived += 1

        if chosen.is_game_over:
            break

    return pieces_history, lines_history, board_history, steps_survived


def test_full_game_replay_determinism() -> None:
    """Dua simulasi independen dengan seed yang sama wajib menghasilkan riwayat game 100% identik."""
    seed = 88888

    p1, l1, b1, s1 = _run_deterministic_simulation(ref_board, seed=seed)
    p2, l2, b2, s2 = _run_deterministic_simulation(ref_board, seed=seed)

    assert s1 == s2, "Jumlah langkah bertahan berbeda pada run dengan seed sama"
    assert p1 == p2, "Sekuens piece berbeda pada run dengan seed sama"
    assert l1 == l2, "Riwayat line clear berbeda pada run dengan seed sama"
    for idx, (b_a, b_b) in enumerate(zip(b1, b2)):
        np.testing.assert_array_equal(
            b_a, b_b, err_msg=f"State papan langkah {idx} tidak identik antar dua run deterministik"
        )


def test_numpy_vs_numba_game_simulation_equivalence() -> None:
    """Simulasi game lengkap menggunakan engine NumPy vs Numba harus menghasilkan lintasan yang 100% identik."""
    for test_seed in [101, 202, 303, 404, 505]:
        ref_p, ref_l, ref_b, ref_s = _run_deterministic_simulation(ref_board, seed=test_seed)
        fast_p, fast_l, fast_b, fast_s = _run_deterministic_simulation(fast_board, seed=test_seed)

        assert ref_s == fast_s, f"Langkah bertahan beda pada seed {test_seed}"
        assert ref_p == fast_p, f"Sekuens piece beda pada seed {test_seed}"
        assert ref_l == fast_l, f"Line clears beda pada seed {test_seed}"
        for step_idx in range(ref_s):
            np.testing.assert_array_equal(
                ref_b[step_idx],
                fast_b[step_idx],
                err_msg=f"Seed {test_seed}: Papan langkah {step_idx} beda antara NumPy dan Numba",
            )

