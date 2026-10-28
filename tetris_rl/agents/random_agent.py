"""Agen acak (Random Agent) untuk batas bawah (lower baseline).

Sesuai Dokumen Proyek Bagian 5.1:
- Memilih aksi secara acak seragam dari seluruh aksi yang sah pada action_mask.
- Menjadi tolok ukur batas bawah performa Tetris RL.
"""

from typing import Any, Dict, Optional, Union
import numpy as np


class RandomAgent:
    """Agen yang memilih aksi legal secara acak seragam."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Mereset kondisi random number generator agen."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(
        self,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
        env: Any = None,
    ) -> int:
        """Memilih satu aksi acak seragam dari himpunan aksi legal.

        Args:
            obs: Observasi lingkungan (opsional, agen acak mengabaikan obs).
            info: Dictionary info lingkungan yang memuat 'action_mask'.
            action_mask: Boolean array mask legalitas aksi (opsional jika info diberikan).

        Returns:
            int: Indeks aksi diskrit yang dipilih.
        """
        mask = action_mask
        if mask is None and info is not None:
            mask = info.get("action_mask")

        if mask is None:
            raise ValueError("RandomAgent membutuhkan 'action_mask' baik via argumen langsung atau dictionary 'info'.")

        valid_actions = np.nonzero(mask)[0]
        if len(valid_actions) == 0:
            return 0

        return int(self.rng.choice(valid_actions))

