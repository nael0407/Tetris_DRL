"""Logika papan Tetris (versi referensi murni NumPy).

Modul ini mengimplementasikan logika inti permainan Tetris:
- Representasi matriks biner papan (default 20 baris x 10 kolom, uint8).
- Deteksi ketinggian kolom (top of column).
- Hard drop piece ke dasar atau tumpukan (landing row).
- Penempatan piece dan deteksi game over (jika menembus batas atas papan).
- Penghapusan baris penuh (1 sampai 4 baris) dan penurunan baris di atasnya.
- Action masking (Discrete(40)) untuk eliminasi rotasi duplikat dan penempatan di luar batas lebar papan.
- Pembuatan afterstate (valid_placements) untuk evaluasi value-based RL.

Modul ini dirancang sebagai acuan kebenaran (ground truth reference) yang diverifikasi
melalui unit test dan dibandingkan secara ketat dengan tetris_rl.board_fast (Numba).
"""

from typing import List, NamedTuple, Tuple
import numpy as np

from tetris_rl.pieces import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    COORDS_TABLE,
    DIMS_TABLE,
    MAX_ROTATIONS,
    NUM_ROTATIONS_TABLE,
    get_piece,
    get_piece_rotation,
)


class PlacementResult(NamedTuple):
    """Hasil dari penempatan suatu piece pada papan."""
    board: np.ndarray        # Board baru hasil penempatan & line clear
    lines_cleared: int       # Jumlah baris yang dihapus (0-4)
    is_game_over: bool       # True jika piece tidak muat (melebihi batas atas row 0)
    landing_row: int         # Posisi baris tempat pojok kiri-atas bounding box piece mendarat


class ValidPlacement(NamedTuple):
    """Informasi afterstate untuk aksi legal."""
    action: int              # Indeks aksi diskrit (0-39)
    rotation: int            # Indeks rotasi (0-3)
    column: int              # Kolom penempatan (0-9)
    next_board: np.ndarray   # Matriks papan afterstate
    lines_cleared: int       # Jumlah baris yang dihapus
    is_game_over: bool       # True jika aksi tersebut berakibat game over


def create_board(height: int = BOARD_HEIGHT_DEFAULT, width: int = BOARD_WIDTH_DEFAULT) -> np.ndarray:
    """Membuat papan Tetris kosong bertipe uint8 (0 = kosong, 1 = terisi)."""
    return np.zeros((height, width), dtype=np.uint8)


def decode_action(action: int, width: int = BOARD_WIDTH_DEFAULT) -> Tuple[int, int]:
    """Mendekode aksi diskrit menjadi pasangan (rotasi, kolom).

    Sesuai Bagian 4.2 Dokumen Proyek:
        rotasi = a // 10
        kolom  = a % 10
    """
    if not (0 <= action < MAX_ROTATIONS * width):
        raise ValueError(f"Action {action} di luar rentang legal [0, {MAX_ROTATIONS * width - 1}]")
    rotation = action // width
    col = action % width
    return rotation, col


def encode_action(rotation: int, col: int, width: int = BOARD_WIDTH_DEFAULT) -> int:
    """Mengenkode pasangan (rotasi, kolom) menjadi aksi diskrit tunggal."""
    if not (0 <= rotation < MAX_ROTATIONS):
        raise ValueError(f"Rotasi {rotation} di luar rentang [0, {MAX_ROTATIONS - 1}]")
    if not (0 <= col < width):
        raise ValueError(f"Kolom {col} di luar rentang [0, {width - 1}]")
    return rotation * width + col


def get_column_tops(board: np.ndarray) -> np.ndarray:
    """Menghitung indeks baris pertama yang terisi untuk setiap kolom.

    Args:
        board: Matriks biner papan (H, W).

    Returns:
        Array 1D int32 panjang W:
        - Jika kolom memiliki blok di baris r, nilai adalah r terkecil (blok teratas).
        - Jika kolom kosong, nilai adalah H (tinggi papan).
    """
    height, width = board.shape
    tops = np.full(width, height, dtype=np.int32)
    for c in range(width):
        filled_rows = np.nonzero(board[:, c])[0]
        if len(filled_rows) > 0:
            tops[c] = filled_rows[0]
    return tops


def compute_landing_row(
    board: np.ndarray,
    coords: np.ndarray,
    col: int,
) -> int:
    """Menghitung baris tempat piece akan mendarat (hard drop dari atas).

    Setiap blok (dr, dc) jatuh vertikal pada kolom (col + dc). Piece akan tertahan
    oleh rintangan teratas pertama pada salah satu kolom yang dilaluinya.

    Args:
        board: Matriks biner papan (H, W).
        coords: Koordinat relatif 4 blok piece (4, 2) [dr, dc].
        col: Kolom penempatan piece (kiri).

    Returns:
        landing_row: Indeks baris pendaratan relatif terhadap pojok kiri atas piece.
                     Bisa bernilai negatif jika piece menabrak tumpukan hingga
                     keluar dari batas atas papan (indikasi Game Over).
    """
    height, width = board.shape
    col_tops = get_column_tops(board)

    # Batas pendaratan ditentukan oleh blok yang paling awal menabrak rintangan atau dasar
    # Untuk tiap blok i: r + dr_i <= col_tops[col + dc_i] - 1
    # r <= col_tops[col + dc_i] - 1 - dr_i
    max_landing_r = height  # Inisialisasi awal di luar papan bawah
    for i in range(4):
        dr = int(coords[i, 0])
        dc = int(coords[i, 1])
        c = col + dc
        allowed_r_for_block = col_tops[c] - 1 - dr
        if allowed_r_for_block < max_landing_r:
            max_landing_r = allowed_r_for_block

    return int(max_landing_r)


def place_piece(
    board: np.ndarray,
    piece_id: int,
    rotation: int,
    col: int,
) -> PlacementResult:
    """Menempatkan piece pada papan dengan hard drop, menghapus baris penuh, dan cek game over.

    Args:
        board: Matriks biner papan (H, W).
        piece_id: ID tetromino (0-6).
        rotation: Indeks rotasi piece.
        col: Kolom penempatan.

    Returns:
        PlacementResult: Berisi board baru, baris terhapus, flag is_game_over, dan landing_row.
    """
    height, width = board.shape
    p_rot = get_piece_rotation(piece_id, rotation)
    coords = p_rot.coords
    p_width = p_rot.width
    p_height = p_rot.height

    if col < 0 or col + p_width > width:
        raise ValueError(
            f"Penempatan kolom {col} tidak valid untuk piece {piece_id} rotasi {rotation} "
            f"dengan lebar {p_width} pada papan lebar {width}"
        )

    landing_row = compute_landing_row(board, coords, col)

    # Deteksi Game Over: Jika bagian mana pun dari piece berada di atas baris 0
    if landing_row < 0:
        # Papan tidak berubah karena piece tidak muat masuk ke papan
        return PlacementResult(
            board=board.copy(),
            lines_cleared=0,
            is_game_over=True,
            landing_row=landing_row,
        )

    new_board = board.copy()

    # Tempatkan 4 blok piece pada new_board
    for i in range(4):
        r = landing_row + coords[i, 0]
        c = col + coords[i, 1]
        new_board[r, c] = 1

    # Cek penghapusan baris (hanya pada rentang baris yang ditempati piece)
    # Rentang baris yang terkena dampak: [landing_row, landing_row + p_height - 1]
    cleared_rows = []
    r_start = landing_row
    r_end = min(landing_row + p_height, height)
    for r in range(r_start, r_end):
        if np.all(new_board[r, :] == 1):
            cleared_rows.append(r)

    num_cleared = len(cleared_rows)

    if num_cleared > 0:
        # Hapus baris yang penuh dan turunkan baris di atasnya
        remaining_rows = [r for r in range(height) if r not in cleared_rows]
        cleared_board = np.zeros_like(new_board)
        # Baris yang tersisa diletakkan di bagian bawah
        cleared_board[num_cleared:, :] = new_board[remaining_rows, :]
        new_board = cleared_board

    return PlacementResult(
        board=new_board,
        lines_cleared=num_cleared,
        is_game_over=False,
        landing_row=landing_row,
    )


def get_action_mask(
    piece_id: int,
    width: int = BOARD_WIDTH_DEFAULT,
) -> np.ndarray:
    """Menghasilkan boolean action mask berukuran 40 (Discrete(40)).

    Sesuai Bagian 4.2 Dokumen Proyek, mask bernilai False pada:
    1. Rotasi duplikat: indeks rotasi di atas jumlah rotasi unik piece.
    2. Keluar papan: penempatan di mana col + width_piece > lebar_papan.

    Returns:
        np.ndarray: Array boolean 1D panjang 40 (True = aksi legal, False = ilegal).
    """
    mask = np.zeros(MAX_ROTATIONS * width, dtype=bool)
    p = get_piece(piece_id)
    num_rots = p.num_rotations

    for rot_idx in range(num_rots):
        p_rot = p.rotations[rot_idx]
        max_valid_col = width - p_rot.width
        for c in range(max_valid_col + 1):
            act = rot_idx * width + c
            mask[act] = True

    return mask


def get_valid_placements(
    board: np.ndarray,
    piece_id: int,
) -> List[ValidPlacement]:
    """Menghitung seluruh afterstate legal untuk piece yang diberikan.

    Fungsi ini digunakan oleh agen value-based (CEM & DQN afterstate)
    untuk mengevaluasi papan hasil penempatan secara menyeluruh.

    Args:
        board: Matriks biner papan saat ini (H, W).
        piece_id: ID piece yang akan dimainkan (0-6).

    Returns:
        List[ValidPlacement]: Daftar afterstate untuk setiap aksi yang lolos action mask.
    """
    height, width = board.shape
    mask = get_action_mask(piece_id, width=width)
    valid_list: List[ValidPlacement] = []

    for action in range(len(mask)):
        if not mask[action]:
            continue

        rot, col = decode_action(action, width=width)
        res = place_piece(board, piece_id, rot, col)
        valid_list.append(
            ValidPlacement(
                action=action,
                rotation=rot,
                column=col,
                next_board=res.board,
                lines_cleared=res.lines_cleared,
                is_game_over=res.is_game_over,
            )
        )

    return valid_list
