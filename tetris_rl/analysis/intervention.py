"""Modul Intervensi Papan Terkontrol untuk Analisis Kausal (intervention.py).

Sesuai Dokumen Proyek Bagian 8:
- Menguji secara kausal apakah fungsi nilai agen bereaksi secara logis terhadap perubahan struktural papan:
  1. Intervensi Lubang (Hole Injection): Menyuntikkan lubang tertutup di bawah tumpukan blok.
     Ekspektasi: Delta V = V(s_hole) - V(s_0) < 0 (nilai harus turun).
  2. Intervensi Ketinggian Kolom (Height Increase): Menaikkan tinggi kolom mendekati batas atas.
     Ekspektasi: Delta V = V(s_height) - V(s_0) < 0 (risiko game over meningkat).
  3. Intervensi Pembuatan Sumur (Well Formation): Membuka sumur 1 kolom vertikal untuk Tetris.
     Ekspektasi: Preferensi aksi saat piece 'I' muncul memilih masuk ke dalam sumur.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch

from tetris_rl.agents.td_afterstate import DQNAfterstateAgent
from tetris_rl.board import get_column_tops
from tetris_rl.features import compute_holes


class InterventionStats(NamedTuple):
    """Statistik efek kausal suatu jenis intervensi terhadap value function."""
    intervention_type: str
    mean_delta_value: float
    std_delta_value: float
    fraction_negative_impact: float
    num_samples: int


def inject_hole(board: np.ndarray, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, bool]:
    """Menyuntikkan 1 lubang buatan di bawah blok yang ada pada papan.

    Returns:
        Tuple[np.ndarray, bool]: (papan termodifikasi, status berhasil)
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = board.shape
    new_board = board.copy()
    tops = get_column_tops(board)

    # Cari kolom-kolom yang memiliki minimal 2 blok bertumpuk
    valid_cols = [c for c in range(W) if tops[c] <= H - 2]
    if not valid_cols:
        return new_board, False

    chosen_c = rng.choice(valid_cols)
    # Suntikkan lubang di bawah baris teratas kolom tersebut
    hole_r = rng.integers(tops[chosen_c] + 1, H)
    if new_board[hole_r, chosen_c] != 0:
        new_board[hole_r, chosen_c] = 0
        return new_board, True

    return new_board, False


def increase_column_height(
    board: np.ndarray,
    num_blocks: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, bool]:
    """Menaikkan tinggi satu kolom secara sengaja dengan menambahkan blok di atasnya."""
    if rng is None:
        rng = np.random.default_rng()

    H, W = board.shape
    new_board = board.copy()
    tops = get_column_tops(board)

    valid_cols = [c for c in range(W) if tops[c] >= num_blocks]
    if not valid_cols:
        return new_board, False

    chosen_c = rng.choice(valid_cols)
    top_r = tops[chosen_c]
    for dr in range(1, num_blocks + 1):
        new_board[top_r - dr, chosen_c] = 1

    return new_board, True


def evaluate_board_value(agent: DQNAfterstateAgent, board: np.ndarray) -> float:
    """Menghitung estimasi nilai skalar V(s) dari agen DQN untuk suatu papan."""
    agent.policy_net.eval()
    feat = agent.state_to_tensor(board, lines_cleared=0)
    with torch.no_grad():
        tensor = torch.from_numpy(feat).unsqueeze(0).to(agent.device)
        val = float(agent.policy_net(tensor).item())
    return val


def run_intervention_analysis(
    agent: DQNAfterstateAgent,
    boards: List[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, InterventionStats]:
    """Mengeksekusi eksperimen intervensi kausal pada sekumpulan sampel papan."""
    if rng is None:
        rng = np.random.default_rng(42)

    delta_v_holes: List[float] = []
    delta_v_heights: List[float] = []

    for b in boards:
        v_base = evaluate_board_value(agent, b)

        # 1. Intervensi Lubang
        b_hole, ok_h = inject_hole(b, rng=rng)
        if ok_h:
            v_hole = evaluate_board_value(agent, b_hole)
            delta_v_holes.append(v_hole - v_base)

        # 2. Intervensi Penambahan Ketinggian (+2 blok)
        b_height, ok_ht = increase_column_height(b, num_blocks=2, rng=rng)
        if ok_ht:
            v_height = evaluate_board_value(agent, b_height)
            delta_v_heights.append(v_height - v_base)

    def summarize_deltas(name: str, deltas: List[float]) -> InterventionStats:
        if not deltas:
            return InterventionStats(name, 0.0, 0.0, 0.0, 0)
        arr = np.array(deltas, dtype=np.float32)
        mean_d = float(np.mean(arr))
        std_d = float(np.std(arr))
        frac_neg = float(np.mean(arr < 0.0))
        return InterventionStats(
            intervention_type=name,
            mean_delta_value=mean_d,
            std_delta_value=std_d,
            fraction_negative_impact=frac_neg,
            num_samples=len(arr),
        )

    return {
        "hole_injection": summarize_deltas("hole_injection", delta_v_holes),
        "height_increase": summarize_deltas("height_increase", delta_v_heights),
    }
