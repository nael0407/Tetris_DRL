"""Tampilan teks (ASCII / Unicode) untuk papan Tetris di terminal.

Modul ini menyediakan fungsi visualisasi berbasis teks untuk:
- Debugging cepat di terminal tanpa kebutuhan grafis (GUI / X11).
- Menampilkan status papan, statistik permainan (skor, baris, langkah),
  serta preview piece yang sedang aktif dan piece berikutnya.
- Overlay posisi piece pada koordinat tertentu sebelum atau sesudah jatuh.

Sesuai Dokumen Proyek Bagian 9, modul rendering terpisah sepenuhnya dari logika
permainan (tidak ada panggilan render di dalam loop pelatihan).
"""

from typing import Optional
import numpy as np

from tetris_rl.pieces import (
    PIECE_ID_TO_NAME,
    PIECE_NAMES,
    get_piece,
    get_piece_rotation,
)


# Representasi grafis piece mini untuk panel samping
_PIECE_MINI_ASCII = {
    "I": [" #### "],
    "O": [" ## ", " ## "],
    "T": [" ### ", "  #  "],
    "S": ["  ## ", " ##  "],
    "Z": [" ##  ", "  ## "],
    "J": [" #   ", " ### "],
    "L": ["   # ", " ### "],
}


def render_board_text(
    board: np.ndarray,
    current_piece_id: Optional[int] = None,
    next_piece_id: Optional[int] = None,
    overlay_piece_id: Optional[int] = None,
    overlay_rotation: int = 0,
    overlay_row: Optional[int] = None,
    overlay_col: Optional[int] = None,
    score: int = 0,
    lines_cleared: int = 0,
    steps: int = 0,
    use_unicode: bool = True,
) -> str:
    """Merender papan Tetris ke dalam format string teks dengan panel statistik di samping.

    Args:
        board: Matriks biner papan (H, W).
        current_piece_id: ID piece yang sedang dimainkan (0-6).
        next_piece_id: ID piece berikutnya pada preview (0-6).
        overlay_piece_id: ID piece yang ingin ditampilkan melayang (ghost/falling).
        overlay_rotation: Indeks rotasi piece overlay.
        overlay_row: Baris pojok kiri atas piece overlay.
        overlay_col: Kolom pojok kiri atas piece overlay.
        score: Skor evaluasi kumulatif.
        lines_cleared: Total baris yang telah dihapus.
        steps: Total langkah/piece yang dimainkan.
        use_unicode: Jika True, gunakan karakter Unicode (┌─┐, ▪). Jika False, gunakan ASCII (+-|#).

    Returns:
        String multi-baris yang siap dicetak di terminal.
    """
    height, width = board.shape

    # Tentukan karakter visual
    if use_unicode:
        c_tl, c_tr, c_bl, c_br = "┌", "┐", "└", "┘"
        c_h, c_v = "──", "│"
        c_empty = " ."
        c_filled = " ■"
        c_overlay = " ▣"
    else:
        c_tl, c_tr, c_bl, c_br = "+", "+", "+", "+"
        c_h, c_v = "--", "|"
        c_empty = " ."
        c_filled = " #"
        c_overlay = " @"

    # Buat salinan representasi grid visual
    grid_chars = [[c_empty for _ in range(width)] for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if board[r, c] != 0:
                grid_chars[r][c] = c_filled

    # Tempel piece overlay jika disediakan
    if (
        overlay_piece_id is not None
        and overlay_row is not None
        and overlay_col is not None
    ):
        p_rot = get_piece_rotation(overlay_piece_id, overlay_rotation)
        for dr, dc in p_rot.coords:
            r = overlay_row + dr
            c = overlay_col + dc
            if 0 <= r < height and 0 <= c < width:
                grid_chars[r][c] = c_overlay

    # Siapkan baris-baris panel samping
    side_info = [
        " TETRIS RL",
        " " + "─" * 16 if use_unicode else " " + "-" * 16,
        f" Score : {score:,}",
        f" Lines : {lines_cleared}",
        f" Steps : {steps}",
        "",
        f" Piece : {PIECE_ID_TO_NAME[current_piece_id] if current_piece_id is not None else '-'}",
        f" Next  : {PIECE_ID_TO_NAME[next_piece_id] if next_piece_id is not None else '-'}",
    ]

    # Tambahkan visual mini untuk Next piece jika ada
    if next_piece_id is not None:
        p_name = PIECE_ID_TO_NAME[next_piece_id]
        side_info.append(" Preview:")
        for line in _PIECE_MINI_ASCII.get(p_name, []):
            side_info.append("  " + line)

    # Susun teks baris per baris
    lines = []
    top_border = c_tl + (c_h * width) + c_tr
    lines.append(top_border)

    for r in range(height):
        row_str = c_v + "".join(grid_chars[r]) + c_v
        extra = side_info[r] if r < len(side_info) else ""
        lines.append(f"{row_str} {extra}")

    bottom_border = c_bl + (c_h * width) + c_br
    lines.append(bottom_border)

    return "\n".join(lines)


def print_board(
    board: np.ndarray,
    current_piece_id: Optional[int] = None,
    next_piece_id: Optional[int] = None,
    **kwargs,
) -> None:
    """Mencetak visualisasi papan ke stdout terminal."""
    print(render_board_text(board, current_piece_id=current_piece_id, next_piece_id=next_piece_id, **kwargs))

