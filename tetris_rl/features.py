"""Ekstraksi fitur buatan tangan (handcrafted features) pada papan Tetris.

Modul ini mengimplementasikan ekstraksi fitur representasi klasik sesuai literatur
(Dellacherie, Szita & Lőrincz 2006, Gabillon et al. 2013) serta Dokumen Proyek Bagian 4.3 & 5.1:
1. Column heights (profil tinggi tiap kolom).
2. Max height (tinggi maksimum tumpukan).
3. Aggregate height (total jumlah ketinggian seluruh kolom).
4. Holes (jumlah lubang: sel kosong yang tertutup blok di atasnya pada kolom yang sama).
5. Bumpiness (kebergelombangan: jumlah selisih mutlak tinggi antar kolom bersebelahan).
6. Lines cleared (jumlah baris yang dihapus).

Menyediakan versi Python murni dan versi Numba JIT untuk evaluasi afterstate super cepat.
"""

from typing import NamedTuple, Tuple
from numba import njit
import numpy as np

from tetris_rl.pieces import BOARD_HEIGHT_DEFAULT, BOARD_WIDTH_DEFAULT


class BoardFeatures(NamedTuple):
    """Koleksi fitur heuristik terstruktur dari suatu konfigurasi papan."""
    column_heights: np.ndarray  # (W,) int32
    max_height: int             # Tinggi maksimum
    aggregate_height: int       # Total tinggi kolom
    holes: int                  # Total lubang
    bumpiness: int              # Kebergelombangan kontur
    lines_cleared: int          # Baris terhapus


@njit(cache=True)
def _compute_features_jit(
    board: np.ndarray,
    lines_cleared: int = 0,
) -> Tuple[np.ndarray, int, int, int, int, int]:
    """JIT kernel: Menghitung seluruh fitur heuristik papan dalam satu pass efisien."""
    H, W = board.shape

    # 1. Hitung tinggi tiap kolom
    col_heights = np.zeros(W, dtype=np.int32)
    holes = 0

    for c in range(W):
        top_found = False
        top_r = H
        for r in range(H):
            if board[r, c] != 0:
                if not top_found:
                    top_found = True
                    top_r = r
            else:
                # Jika sel kosong dan sudah ada blok di atasnya -> ini lubang (hole)
                if top_found:
                    holes += 1

        if top_found:
            col_heights[c] = H - top_r

    # 2. Max height & Aggregate height
    max_h = 0
    agg_h = 0
    for c in range(W):
        h = col_heights[c]
        agg_h += h
        if h > max_h:
            max_h = h

    # 3. Bumpiness (selisih mutlak tinggi kolom bersebelahan)
    bumpiness = 0
    for c in range(W - 1):
        diff = col_heights[c] - col_heights[c + 1]
        bumpiness += abs(diff)

    return col_heights, max_h, agg_h, holes, bumpiness, lines_cleared


def compute_column_heights(board: np.ndarray) -> np.ndarray:
    """Menghitung tinggi tiap kolom (0 sampai H)."""
    col_heights, _, _, _, _, _ = _compute_features_jit(board, 0)
    return col_heights


def compute_holes(board: np.ndarray) -> int:
    """Menghitung total sel kosong yang tertutup blok di atasnya."""
    _, _, _, holes, _, _ = _compute_features_jit(board, 0)
    return holes


def compute_bumpiness(board: np.ndarray) -> int:
    """Menghitung total selisih mutlak tinggi antara kolom-kolom bersebelahan."""
    _, _, _, _, bumpiness, _ = _compute_features_jit(board, 0)
    return bumpiness


def compute_aggregate_height(board: np.ndarray) -> int:
    """Menghitung jumlah tinggi seluruh kolom."""
    _, _, agg_h, _, _, _ = _compute_features_jit(board, 0)
    return agg_h


def compute_max_height(board: np.ndarray) -> int:
    """Menghitung tinggi tertinggi di antara seluruh kolom."""
    _, max_h, _, _, _, _ = _compute_features_jit(board, 0)
    return max_h


def get_board_features(board: np.ndarray, lines_cleared: int = 0) -> BoardFeatures:
    """Mengembalikan objek BoardFeatures berisi seluruh metrik heuristik."""
    col_h, max_h, agg_h, holes, bump, lc = _compute_features_jit(board, lines_cleared)
    return BoardFeatures(
        column_heights=col_h,
        max_height=int(max_h),
        aggregate_height=int(agg_h),
        holes=int(holes),
        bumpiness=int(bump),
        lines_cleared=int(lc),
    )


def extract_feature_vector(
    board: np.ndarray,
    lines_cleared: int = 0,
    normalize: bool = False,
) -> np.ndarray:
    """Mengekstraksi vektor 1D float32 fitur buatan tangan untuk observasi agen RL.

    Format vektor (panjang W + 5):
    [col_0, col_1, ..., col_{W-1}, max_height, aggregate_height, holes, bumpiness, lines_cleared]

    Args:
        board: Matriks biner papan (H, W).
        lines_cleared: Baris terhapus pada langkah terkait.
        normalize: Jika True, menormalisasi fitur ke rentang aproksimasi [0, 1].

    Returns:
        np.ndarray: Vektor 1D float32 berukuran (W + 5).
    """
    H, W = board.shape
    col_h, max_h, agg_h, holes, bump, lc = _compute_features_jit(board, lines_cleared)

    feat_dim = W + 5
    vec = np.empty(feat_dim, dtype=np.float32)

    if normalize:
        # Normalisasi dengan tinggi maksimum dan lebar papan
        vec[:W] = col_h / float(H)
        vec[W] = float(max_h) / float(H)
        vec[W + 1] = float(agg_h) / float(H * W)
        vec[W + 2] = float(holes) / float(H * W)
        vec[W + 3] = float(bump) / float(H * (W - 1))
        vec[W + 4] = float(lc) / 4.0
    else:
        vec[:W] = col_h.astype(np.float32)
        vec[W] = float(max_h)
        vec[W + 1] = float(agg_h)
        vec[W + 2] = float(holes)
        vec[W + 3] = float(bump)
        vec[W + 4] = float(lc)

    return vec

