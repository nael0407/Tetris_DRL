"""Logika papan Tetris terakselerasi Numba (High-Performance Engine).

Modul ini mengimplementasikan logika penempatan, deteksi landing row, hard drop,
dan penghapusan baris menggunakan compiler Just-In-Time (JIT) Numba.
Fungsi-fungsi di modul ini dioptimasi untuk kecepatan tinggi saat pelatihan
agen RL (DQN afterstate dan evaluasi kandidat CEM).

Kesetaraan perilaku (behavioral equivalence) modul ini terhadap versi referensi
(tetris_rl.board) diverifikasi secara ketat melalui test_board_equiv.py.
"""

from typing import List, Tuple
import numba
from numba import njit
import numpy as np

from tetris_rl.board import (
    PlacementResult,
    ValidPlacement,
    decode_action,
    encode_action,
    get_action_mask as _ref_get_action_mask,
)
from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    COORDS_TABLE,
    DIMS_TABLE,
    MAX_ROTATIONS,
    NUM_ROTATIONS_TABLE,
)


@njit(cache=True)
def _get_column_tops_jit(board: np.ndarray) -> np.ndarray:
    """JIT kernel: Menghitung baris pertama yang terisi pada tiap kolom."""
    H, W = board.shape
    tops = np.full(W, H, dtype=np.int32)
    for c in range(W):
        for r in range(H):
            if board[r, c] != 0:
                tops[c] = r
                break
    return tops


@njit(cache=True)
def _compute_landing_row_jit(tops: np.ndarray, coords: np.ndarray, col: int) -> int:
    """JIT kernel: Menghitung baris pendaratan piece berdasarkan top of column."""
    max_landing_r = 1000000
    for i in range(4):
        dr = coords[i, 0]
        dc = coords[i, 1]
        c = col + dc
        allowed_r = tops[c] - 1 - dr
        if allowed_r < max_landing_r:
            max_landing_r = allowed_r
    return max_landing_r


@njit(cache=True)
def _place_and_clear_jit(
    board: np.ndarray,
    coords: np.ndarray,
    landing_row: int,
    col: int,
) -> Tuple[np.ndarray, int]:
    """JIT kernel: Menempatkan blok piece dan menghapus baris penuh secara efisien."""
    H, W = board.shape
    new_board = np.empty((H, W), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            new_board[r, c] = board[r, c]

    # Pasang 4 sel piece
    for i in range(4):
        r = landing_row + coords[i, 0]
        c = col + coords[i, 1]
        new_board[r, c] = 1

    # Penghapusan baris dua penunjuk (in-place row downward compaction)
    write_row = H - 1
    lines_cleared = 0

    for r in range(H - 1, -1, -1):
        is_full = True
        for c in range(W):
            if new_board[r, c] == 0:
                is_full = False
                break

        if is_full:
            lines_cleared += 1
        else:
            if write_row != r:
                for c in range(W):
                    new_board[write_row, c] = new_board[r, c]
            write_row -= 1

    # Isi sisa baris teratas dengan 0
    for r in range(write_row, -1, -1):
        for c in range(W):
            new_board[r, c] = 0

    return new_board, lines_cleared


@njit(cache=True)
def _place_piece_jit(
    board: np.ndarray,
    piece_id: int,
    rotation: int,
    col: int,
    coords_table: np.ndarray,
    dims_table: np.ndarray,
) -> Tuple[np.ndarray, int, bool, int]:
    """JIT kernel: Eksekusi lengkap penempatan piece."""
    coords = coords_table[piece_id, rotation]
    tops = _get_column_tops_jit(board)
    landing_row = _compute_landing_row_jit(tops, coords, col)

    if landing_row < 0:
        return board.copy(), 0, True, landing_row

    new_board, lines_cleared = _place_and_clear_jit(board, coords, landing_row, col)
    return new_board, lines_cleared, False, landing_row


@njit(cache=True)
def _compute_afterstates_jit(
    board: np.ndarray,
    piece_id: int,
    coords_table: np.ndarray,
    dims_table: np.ndarray,
    num_rotations_table: np.ndarray,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """JIT kernel: Menghitung afterstate untuk seluruh penempatan legal secara cepat.

    Returns:
        actions: (N,) int32
        next_boards: (N, H, W) uint8
        lines_cleared: (N,) int32
        game_over_flags: (N,) bool
        count: N (jumlah aksi legal)
    """
    H, W = board.shape
    num_rots = num_rotations_table[piece_id]

    # Alokasi buffer maksimum (maks 40 aksi)
    max_actions = 40
    out_actions = np.empty(max_actions, dtype=np.int32)
    out_boards = np.empty((max_actions, H, W), dtype=np.uint8)
    out_cleared = np.empty(max_actions, dtype=np.int32)
    out_game_over = np.empty(max_actions, dtype=np.bool_)

    count = 0
    for rot in range(num_rots):
        p_width = dims_table[piece_id, rot, 1]
        max_col = width - p_width
        for c in range(max_col + 1):
            act = rot * width + c
            nb, lc, go, lr = _place_piece_jit(
                board, piece_id, rot, c, coords_table, dims_table
            )
            out_actions[count] = act
            out_boards[count] = nb
            out_cleared[count] = lc
            out_game_over[count] = go
            count += 1

    return out_actions[:count], out_boards[:count], out_cleared[:count], out_game_over[:count], count


# --- Public Python Wrapper API (Kompatibel dengan tetris_rl.board) ---


def create_board(height: int = BOARD_HEIGHT_DEFAULT, width: int = BOARD_WIDTH_DEFAULT) -> np.ndarray:
    """Membuat papan Tetris kosong bertipe uint8."""
    return np.zeros((height, width), dtype=np.uint8)


def get_column_tops(board: np.ndarray) -> np.ndarray:
    """Wrapper Numba untuk get_column_tops."""
    return _get_column_tops_jit(board)


def compute_landing_row(board: np.ndarray, coords: np.ndarray, col: int) -> int:
    """Wrapper Numba untuk compute_landing_row."""
    tops = _get_column_tops_jit(board)
    return int(_compute_landing_row_jit(tops, coords, col))


def place_piece(
    board: np.ndarray,
    piece_id: int,
    rotation: int,
    col: int,
) -> PlacementResult:
    """Wrapper Numba untuk penempatan piece pada papan."""
    width = board.shape[1]
    p_width = DIMS_TABLE[piece_id, rotation, 1]
    if col < 0 or col + p_width > width:
        raise ValueError(
            f"Penempatan kolom {col} tidak valid untuk piece {piece_id} rotasi {rotation} "
            f"dengan lebar {p_width} pada papan lebar {width}"
        )

    new_board, lines_cleared, is_game_over, landing_row = _place_piece_jit(
        board, piece_id, rotation, col, COORDS_TABLE, DIMS_TABLE
    )

    return PlacementResult(
        board=new_board,
        lines_cleared=int(lines_cleared),
        is_game_over=bool(is_game_over),
        landing_row=int(landing_row),
    )


def get_action_mask(piece_id: int, width: int = BOARD_WIDTH_DEFAULT) -> np.ndarray:
    """Mengembalikan boolean action mask 40 aksi."""
    return _ref_get_action_mask(piece_id, width=width)


def get_valid_placements(board: np.ndarray, piece_id: int) -> List[ValidPlacement]:
    """Wrapper Numba untuk menghitung afterstate seluruh penempatan legal."""
    height, width = board.shape
    actions, boards, cleared, game_overs, count = _compute_afterstates_jit(
        board,
        piece_id,
        COORDS_TABLE,
        DIMS_TABLE,
        NUM_ROTATIONS_TABLE,
        width,
    )

    valid_list: List[ValidPlacement] = []
    for i in range(count):
        act = int(actions[i])
        rot, col = decode_action(act, width=width)
        valid_list.append(
            ValidPlacement(
                action=act,
                rotation=rot,
                column=col,
                next_board=boards[i],
                lines_cleared=int(cleared[i]),
                is_game_over=bool(game_overs[i]),
            )
        )
    return valid_list
