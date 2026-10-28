"""Unit tests untuk modul logika papan Tetris (tetris_rl.board).

Memverifikasi:
- Pembuatan papan awal kosong dengan ukuran default dan kustom.
- Enkode dan dekode aksi diskrit (rotasi, kolom).
- Perhitungan tinggi/top kolom secara akurat.
- Pendaratan piece (hard drop) pada papan kosong dan bertumpuk.
- Penghapusan 1, 2, 3, hingga 4 baris (Tetris) sekaligus.
- Penurunan baris di atas baris yang terhapus dan pengisian baris kosong di atas.
- Deteksi kondisi Game Over jika piece melebihi batas atas papan (row 0).
- Action masking dan penghitungan afterstate (valid_placements).
"""

import numpy as np
import pytest

from tetris_rl.board import (
    compute_landing_row,
    create_board,
    decode_action,
    encode_action,
    get_action_mask,
    get_column_tops,
    get_valid_placements,
    place_piece,
)
from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    PIECE_NAME_TO_ID,
    get_piece,
    get_piece_rotation,
)


def test_create_board() -> None:
    """Memastikan papan default berukuran 20x10 uint8 dan berisi nol."""
    board = create_board()
    assert board.shape == (BOARD_HEIGHT_DEFAULT, BOARD_WIDTH_DEFAULT)
    assert board.dtype == np.uint8
    assert np.all(board == 0)

    # Custom size
    board_custom = create_board(height=12, width=8)
    assert board_custom.shape == (12, 8)
    assert np.all(board_custom == 0)


def test_encode_decode_action() -> None:
    """Memastikan enkode dan dekode aksi bersifat bijektif."""
    width = 10
    for rot in range(4):
        for col in range(width):
            action = encode_action(rot, col, width=width)
            assert action == rot * width + col
            dec_rot, dec_col = decode_action(action, width=width)
            assert dec_rot == rot
            assert dec_col == col

    # Error checking
    with pytest.raises(ValueError):
        decode_action(-1)
    with pytest.raises(ValueError):
        decode_action(40)
    with pytest.raises(ValueError):
        encode_action(4, 0)
    with pytest.raises(ValueError):
        encode_action(0, 10)


def test_get_column_tops() -> None:
    """Memastikan get_column_tops menghitung indeks baris teratas dengan benar."""
    board = create_board(height=10, width=5)
    # Papan kosong -> tops harus bernilai H (10)
    tops = get_column_tops(board)
    np.testing.assert_array_equal(tops, np.full(5, 10))

    # Beri blok di kolom 0 baris 7, kolom 2 baris 4 dan 6, kolom 4 baris 0
    board[7, 0] = 1
    board[4, 2] = 1
    board[6, 2] = 1
    board[0, 4] = 1

    expected_tops = np.array([7, 10, 4, 10, 0], dtype=np.int32)
    np.testing.assert_array_equal(get_column_tops(board), expected_tops)


def test_landing_row_on_empty_board() -> None:
    """Piece harus mendarat tepat di dasar papan kosong."""
    board = create_board(height=20, width=10)
    p_i = get_piece("I")

    # I horizontal: h=1, coords dr=0. Landing row harus di 20 - 1 = 19
    coords_h = p_i.rotations[0].coords
    landing_h = compute_landing_row(board, coords_h, col=3)
    assert landing_h == 19

    # I vertikal: h=4, coords dr=0,1,2,3. Landing row harus di 20 - 4 = 16
    coords_v = p_i.rotations[1].coords
    landing_v = compute_landing_row(board, coords_v, col=3)
    assert landing_v == 16


def test_stacking_pieces() -> None:
    """Piece yang diletakkan di atas piece lain harus menumpuk tanpa tumpang tindih."""
    board = create_board(height=20, width=10)
    id_o = PIECE_NAME_TO_ID["O"]

    # Letakkan piece O pertama di kolom 0 (mendarat di baris 18..19)
    res1 = place_piece(board, piece_id=id_o, rotation=0, col=0)
    assert not res1.is_game_over
    assert res1.landing_row == 18
    assert res1.lines_cleared == 0
    assert np.sum(res1.board) == 4
    assert np.all(res1.board[18:20, 0:2] == 1)

    # Letakkan piece O kedua di atasnya (kolom 0 lagi -> mendarat di baris 16..17)
    res2 = place_piece(res1.board, piece_id=id_o, rotation=0, col=0)
    assert not res2.is_game_over
    assert res2.landing_row == 16
    assert res2.lines_cleared == 0
    assert np.sum(res2.board) == 8
    assert np.all(res2.board[16:20, 0:2] == 1)


def test_single_line_clear() -> None:
    """Pengujian penghapusan 1 baris penuh."""
    board = create_board(height=20, width=10)
    # Isi baris 19 kecuali kolom 0, 1, 2, 3
    board[19, 4:10] = 1
    # Isi juga baris 18 sebagian untuk cek penurunan
    board[18, 5] = 1

    # Tempatkan I horizontal (lebar 4) di kolom 0
    id_i = PIECE_NAME_TO_ID["I"]
    res = place_piece(board, piece_id=id_i, rotation=0, col=0)

    assert not res.is_game_over
    assert res.lines_cleared == 1
    # Baris 19 terhapus, blok di baris 18 kolom 5 harus turun ke baris 19 kolom 5
    assert res.board[19, 5] == 1
    assert res.board[18, 5] == 0
    assert np.sum(res.board) == 1  # Hanya tersisa 1 blok


def test_multi_line_clear_and_tetris() -> None:
    """Pengujian penghapusan 2, 3, dan 4 baris (Tetris)."""
    id_i = PIECE_NAME_TO_ID["I"]

    # 1. Dua baris sekaligus dengan O piece (2x2)
    board2 = create_board(height=20, width=10)
    board2[18:20, 2:10] = 1
    res_o = place_piece(board2, piece_id=PIECE_NAME_TO_ID["O"], rotation=0, col=0)
    assert res_o.lines_cleared == 2
    assert np.sum(res_o.board) == 0

    # 2. Empat baris (Tetris) dengan I vertikal (4x1)
    board4 = create_board(height=20, width=10)
    # Isi baris 16..19 pada kolom 1..9 (lebar 9 penuh, kolom 0 sumur kosong)
    board4[16:20, 1:10] = 1
    # Beri penanda di baris 15 kolom 3 (harus turun 4 tingkat ke baris 19)
    board4[15, 3] = 1

    # Drop I vertikal (rotasi 1, lebar 1) di sumur kolom 0
    res_tetris = place_piece(board4, piece_id=id_i, rotation=1, col=0)
    assert res_tetris.lines_cleared == 4
    assert not res_tetris.is_game_over
    # Baris 15 turun 4 langkah menjadi baris 19
    assert res_tetris.board[19, 3] == 1
    assert np.sum(res_tetris.board) == 1


def test_game_over_detection() -> None:
    """Piece yang menumpuk hingga meluap ke atas baris 0 harus memicu game over."""
    board = create_board(height=10, width=10)
    id_i = PIECE_NAME_TO_ID["I"]

    # Tumpuk I vertikal (tinggi 4):
    # 1st: baris 6..9 (landing=6)
    r1 = place_piece(board, id_i, rotation=1, col=0)
    assert not r1.is_game_over
    assert r1.landing_row == 6

    # 2nd: baris 2..5 (landing=2)
    r2 = place_piece(r1.board, id_i, rotation=1, col=0)
    assert not r2.is_game_over
    assert r2.landing_row == 2

    # 3rd: baris -2..1 (landing=-2, menembus atas) -> Game Over!
    r3 = place_piece(r2.board, id_i, rotation=1, col=0)
    assert r3.is_game_over
    assert r3.landing_row < 0
    # Pada game over, board yang dikembalikan tidak rusak
    np.testing.assert_array_equal(r3.board, r2.board)


def test_action_mask() -> None:
    """Memverifikasi action mask memblokir aksi keluar batas dan rotasi duplikat."""
    # O piece: hanya 1 rotasi unik (rot 0), lebar 2 -> kolom valid 0..8 (9 kolom)
    mask_o = get_action_mask(PIECE_NAME_TO_ID["O"], width=10)
    assert np.sum(mask_o) == 9
    for act in range(40):
        rot = act // 10
        col = act % 10
        if rot == 0 and col <= 8:
            assert mask_o[act] is np.True_
        else:
            assert mask_o[act] is np.False_

    # I piece:
    # rot 0 (lebar 4): kolom 0..6 (7 valid)
    # rot 1 (lebar 1): kolom 0..9 (10 valid)
    # rot 2, 3: rotasi duplikat (0 valid)
    # Total valid = 7 + 10 = 17
    mask_i = get_action_mask(PIECE_NAME_TO_ID["I"], width=10)
    assert np.sum(mask_i) == 17


def test_valid_placements_count() -> None:
    """get_valid_placements harus mengembalikan afterstate tepat sebanyak aksi legal pada mask."""
    board = create_board()
    for name in ["I", "O", "T", "S", "Z", "J", "L"]:
        p_id = PIECE_NAME_TO_ID[name]
        mask = get_action_mask(p_id, width=10)
        expected_count = int(np.sum(mask))

        placements = get_valid_placements(board, p_id)
        assert len(placements) == expected_count

        for p in placements:
            assert mask[p.action] is np.True_
            assert not p.is_game_over
            assert p.next_board.shape == board.shape
