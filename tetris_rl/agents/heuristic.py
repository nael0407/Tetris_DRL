"""Agen Heuristik Berbobot Linear (Heuristic Agent).

Sesuai Dokumen Proyek Bagian 5.1:
- Memilih penempatan piece yang memaksimalkan fungsi nilai linear terhadap fitur afterstate:
  V(s') = w_1 * aggregate_height + w_2 * lines_cleared + w_3 * holes + w_4 * bumpiness
- Bobot default diinisialisasi dari literatur standar Tetris AI (Dellacherie / Fahey / Szita & Lőrincz).
- Bobot dapat diperbarui atau dimuat dari hasil optimasi Cross-Entropy Method (CEM).
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from tetris_rl.board import ValidPlacement
from tetris_rl.features import (
    compute_aggregate_height,
    compute_bumpiness,
    compute_holes,
    get_board_features,
)

# Bobot heuristik klasik Tetris standar (Dellacherie / Fahey normalized)
DEFAULT_HEURISTIC_WEIGHTS = np.array(
    [-0.510066, 0.760666, -0.356630, -0.184483],
    dtype=np.float32,
)


class HeuristicAgent:
    """Agen penempatan Tetris berbasis evaluasi afterstate dengan fungsi nilai linear."""

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        name: str = "Dellacherie-Heuristic",
    ) -> None:
        """Inisialisasi agen heuristik.

        Args:
            weights: Vektor bobot 4 dimensi [w_height, w_lines, w_holes, w_bumpiness].
                     Jika None, gunakan bobot standar literatur.
            name: Nama pengenal agen.
        """
        self.name = name
        if weights is None:
            self.weights = DEFAULT_HEURISTIC_WEIGHTS.copy()
        else:
            self.weights = np.asarray(weights, dtype=np.float32)

        if len(self.weights) not in (4, 200):
            raise ValueError(
                f"Vektor bobot heuristik harus berdimensi 4 (fitur) atau 200 (papan mentah 20x10), didapat {len(self.weights)}"
            )

    def evaluate_placement(self, placement: ValidPlacement) -> float:
        """Menghitung nilai evaluasi linear untuk suatu afterstate penempatan."""
        if placement.is_game_over:
            return -1e9  # Penalti ekstrim untuk mencegah game over

        # Mode papan mentah (200 piksel)
        if len(self.weights) == 200:
            board_flat = placement.next_board.flatten().astype(np.float32)
            # Nilai dot product + bonus pembersihan baris
            score = float(np.dot(self.weights, board_flat) + placement.lines_cleared * 100.0)
            return score

        # Ekstraksi fitur afterstate klasik (4 fitur)
        board = placement.next_board
        holes = float(compute_holes(board))
        bump = float(compute_bumpiness(board))
        agg_h = float(compute_aggregate_height(board))
        lines = float(placement.lines_cleared)

        # Nilai linear: w0 * agg_h + w1 * lines + w2 * holes + w3 * bump
        score = (
            self.weights[0] * agg_h
            + self.weights[1] * lines
            + self.weights[2] * holes
            + self.weights[3] * bump
        )
        return float(score)

    def act(
        self,
        env: Any = None,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Memilih aksi terbaik dengan mengevaluasi seluruh afterstate legal pada lingkungan."""
        placements: List[ValidPlacement] = env.valid_placements()
        if not placements:
            return 0

        best_action = placements[0].action
        best_val = -1e12

        # Evaluasi deterministik dengan pemecah seri (tie-breaker) yang stabil
        for p in placements:
            val = self.evaluate_placement(p)
            # Tie-breaking: jika nilai identik, dahulukan landing_row terendah (paling dasar)
            if val > best_val or (val == best_val and p.landing_row > placements[0].landing_row):
                best_val = val
                best_action = p.action

        return best_action

