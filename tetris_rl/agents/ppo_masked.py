"""Agen Policy Gradient: MaskablePPO dengan Action Masking.

Sesuai Dokumen Proyek Bagian 5.2 dan 6 (E3 & E5):
- Mengintegrasikan sb3-contrib.MaskablePPO dengan action_masks() dari TetrisEnv.
- Mengeliminasi penempatan ilegal (rotasi/posisi overflow) secara langsung dari distribusi kebijakan pi(a|s).
- Mendukung mode observasi raw (papan biner), features (fitur Dellacherie), maupun combined.
"""

import os
from typing import Any, Callable, Dict, Optional, Union
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from tetris_rl.env import TetrisEnv


def tetris_mask_fn(env: gym.Env) -> np.ndarray:
    """Fungsi ekstraksi action mask untuk ActionMasker wrapper sb3-contrib."""
    # Menembus wrapper jika env dibungkus
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_masks"):
        return unwrapped.action_masks()
    raise AttributeError("Environment tidak memiliki metode 'action_masks()'.")


def make_masked_tetris_env(
    board_height: int = 20,
    board_width: int = 10,
    obs_type: str = "raw",
    reward_type: str = "R0",
    seed: Optional[int] = None,
) -> gym.Env:
    """Membangun environment TetrisEnv yang dibungkus ActionMasker untuk MaskablePPO."""
    raw_env = TetrisEnv(
        board_height=board_height,
        board_width=board_width,
        obs_type=obs_type,
        reward_type=reward_type,
        use_numba=True,
    )
    if seed is not None:
        raw_env.reset(seed=seed)
    return ActionMasker(raw_env, tetris_mask_fn)


class MaskedPPOAgent:
    """Wrapper Agen MaskablePPO untuk antarmuka evaluasi dan inferensi konsisten."""

    def __init__(
        self,
        model: Optional[MaskablePPO] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        if model is not None:
            self.model = model
        elif model_path is not None and os.path.exists(model_path):
            self.model = MaskablePPO.load(model_path, device=device)
        else:
            self.model = None

    def act(
        self,
        env: Any = None,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> int:
        """Memilih aksi menggunakan distribusi kebijakan ber-masking."""
        if self.model is None:
            raise RuntimeError("Model MaskablePPO belum diinisialisasi atau dimuat.")

        if obs is None:
            raise ValueError("MaskedPPOAgent membutuhkan 'obs' untuk inferensi kebijakan.")

        # Ambil action mask dari argumen, info, atau env
        mask = action_mask
        if mask is None and info is not None:
            mask = info.get("action_mask")
        if mask is None and env is not None:
            mask = tetris_mask_fn(env)

        if mask is None:
            raise ValueError("MaskedPPOAgent membutuhkan 'action_mask' untuk inferensi.")

        action, _states = self.model.predict(
            obs,
            action_masks=mask,
            deterministic=deterministic,
        )
        return int(action)

    def save(self, path: str) -> None:
        """Menyimpan model MaskablePPO ke file zip."""
        if self.model is None:
            raise RuntimeError("Tidak ada model untuk disimpan.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save(path)

    def load(self, path: str, device: str = "auto") -> None:
        """Memuat model MaskablePPO dari file zip."""
        self.model = MaskablePPO.load(path, device=device)

