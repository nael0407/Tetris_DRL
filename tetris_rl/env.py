"""Antarmuka lingkungan Tetris berbasis Gymnasium (TetrisEnv).

Modul ini mengimplementasikan TetrisEnv yang sepenuhnya kompatibel dengan spesifikasi
Gymnasium dan antarmuka MaskablePPO (sb3-contrib):
- Ruang aksi diskrit: Discrete(40) dengan action masking (info["action_mask"]).
- Tiga representasi observasi: 'raw' (papan biner + one-hot piece), 'features' (fitur buatan tangan + one-hot), dan 'combined'.
- Empat skema reward pelatihan: R0 (skor baris saja), R1 (R0 + bonus bertahan), R2 (R1 - penalti lubang/tinggi), dan R3 (potential-based shaping).
- Skor evaluasi permainan (game score: 100/300/500/800) disimpan terpisah pada info["score"].
- Penanda terminasi terstandarisasi: terminated (game over) dan truncated (mencapai batas langkah).
- Pilihan mesin simulasi: Numba JIT (cepat, default) atau NumPy murni.

Memenuhi kepatuhan penuh terhadap gymnasium.utils.env_checker.check_env.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import tetris_rl.board as ref_board
import tetris_rl.board_fast as fast_board
from tetris_rl.board import PlacementResult, ValidPlacement, decode_action
from tetris_rl.features import (
    compute_aggregate_height,
    compute_bumpiness,
    compute_holes,
    compute_max_height,
    extract_feature_vector,
    get_board_features,
)
from tetris_rl.pieces import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    MAX_ROTATIONS,
    NUM_PIECES,
    PieceGenerator,
    create_piece_generator,
    get_piece,
)
from tetris_rl.render_pygame import PygameRenderer
from tetris_rl.render_text import render_board_text

# Poin skor resmi permainan Tetris (Dokumen Proyek Bagian 3.3)
LINE_CLEAR_POINTS: Dict[int, int] = {
    0: 0,
    1: 100,
    2: 300,
    3: 500,
    4: 800,
}


class TetrisEnv(gym.Env):
    """Lingkungan Tetris RL berbasis Gymnasium."""

    metadata = {
        "render_modes": ["ansi", "rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        board_height: int = BOARD_HEIGHT_DEFAULT,
        board_width: int = BOARD_WIDTH_DEFAULT,
        obs_type: str = "raw",
        reward_type: str = "R0",
        piece_generator_type: str = "uniform",
        max_steps: Optional[int] = 10000,
        render_mode: Optional[str] = None,
        use_numba: bool = True,
    ) -> None:
        """Inisialisasi TetrisEnv.

        Args:
            board_height: Tinggi papan (default 20).
            board_width: Lebar papan (default 10).
            obs_type: Jenis observasi ('raw', 'features', 'combined').
            reward_type: Skema reward pelatihan ('R0', 'R1', 'R2', 'R3').
            piece_generator_type: Generator tetromino ('uniform' atau '7-bag').
            max_steps: Batas maksimal langkah piece per episode (default 10.000).
            render_mode: Mode rendering ('ansi', 'rgb_array', 'human', atau None).
            use_numba: Jika True, gunakan mesin Numba berkecepatan tinggi.
        """
        super().__init__()

        self.board_height = board_height
        self.board_width = board_width
        self.obs_type = obs_type.lower()
        self.reward_type = reward_type.upper()
        self.piece_generator_type = piece_generator_type.lower()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_numba = use_numba

        # Pilih modul mesin penempatan
        self.engine = fast_board if self.use_numba else ref_board

        # Ruang aksi diskrit: 4 rotasi x board_width (default 40 aksi)
        self.action_space = spaces.Discrete(MAX_ROTATIONS * self.board_width)

        # Definisikan ruang observasi
        self.num_piece_types = NUM_PIECES  # 7 jenis piece
        raw_size = self.board_height * self.board_width + (self.num_piece_types * 2)
        features_size = (self.board_width + 5) + (self.num_piece_types * 2)

        if self.obs_type == "raw":
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(raw_size,),
                dtype=np.float32,
            )
        elif self.obs_type == "features":
            self.observation_space = spaces.Box(
                low=0.0,
                high=float(max(self.board_height * self.board_width, 100)),
                shape=(features_size,),
                dtype=np.float32,
            )
        elif self.obs_type == "combined":
            combined_size = (
                self.board_height * self.board_width
                + (self.board_width + 5)
                + (self.num_piece_types * 2)
            )
            self.observation_space = spaces.Box(
                low=0.0,
                high=float(max(self.board_height * self.board_width, 100)),
                shape=(combined_size,),
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"obs_type '{self.obs_type}' tidak dikenal. Pilihan: 'raw', 'features', 'combined'."
            )

        # Generator piece
        self.piece_generator: PieceGenerator = create_piece_generator(
            self.piece_generator_type
        )

        # State internal
        self.board: np.ndarray = self.engine.create_board(self.board_height, self.board_width)
        self.current_piece: int = 0
        self.next_piece: int = 0
        self.score: int = 0
        self.lines_cleared: int = 0
        self.steps: int = 0
        self.prev_potential: float = 0.0

        # Renderer grafis (dibuat lazily jika render_mode aktif)
        self._renderer: Optional[PygameRenderer] = None

    def action_masks(self) -> np.ndarray:
        """Helper untuk integrasi langsung dengan sb3-contrib MaskablePPO."""
        return self.engine.get_action_mask(self.current_piece, width=self.board_width)

    def _get_obs(self) -> np.ndarray:
        """Membangun vektor observasi sesuai dengan obs_type."""
        # One-hot encoding untuk current piece dan next piece
        curr_onehot = np.zeros(self.num_piece_types, dtype=np.float32)
        curr_onehot[self.current_piece] = 1.0

        next_onehot = np.zeros(self.num_piece_types, dtype=np.float32)
        next_onehot[self.next_piece] = 1.0

        if self.obs_type == "raw":
            board_flat = self.board.flatten().astype(np.float32)
            return np.concatenate([board_flat, curr_onehot, next_onehot])

        feat_vec = extract_feature_vector(self.board, lines_cleared=0, normalize=False)

        if self.obs_type == "features":
            return np.concatenate([feat_vec, curr_onehot, next_onehot])

        if self.obs_type == "combined":
            board_flat = self.board.flatten().astype(np.float32)
            return np.concatenate([board_flat, feat_vec, curr_onehot, next_onehot])

        raise RuntimeError(f"Tipe observasi {self.obs_type} tidak tertangani")

    def _compute_potential(self, board: np.ndarray) -> float:
        """Menghitung nilai potensial papan untuk reward shaping R3.

        Berdasarkan bobot heuristik Dellacherie standar (bernilai negatif):
        Phi(s) = - (4.0 * holes + 1.0 * bumpiness + 0.5 * aggregate_height)
        """
        holes = compute_holes(board)
        bump = compute_bumpiness(board)
        agg_h = compute_aggregate_height(board)
        return float(- (4.0 * holes + 1.0 * bump + 0.5 * agg_h))

    def _compute_reward(
        self,
        lines_cleared: int,
        is_game_over: bool,
        board_before: np.ndarray,
        board_after: np.ndarray,
    ) -> float:
        """Menghitung reward pelatihan sesuai skema yang dipilih (R0 sampai R3)."""
        game_points = float(LINE_CLEAR_POINTS.get(lines_cleared, 0))

        if self.reward_type == "R0":
            # Hanya poin baris terhapus
            return game_points

        elif self.reward_type == "R1":
            # R0 + bonus bertahan per piece
            survival_bonus = 1.0 if not is_game_over else 0.0
            penalty = -10.0 if is_game_over else 0.0
            return game_points + survival_bonus + penalty

        elif self.reward_type == "R2":
            # R1 - penalti lubang - penalti tinggi tumpukan
            survival_bonus = 1.0 if not is_game_over else 0.0
            penalty = -20.0 if is_game_over else 0.0
            holes = compute_holes(board_after)
            max_h = compute_max_height(board_after)
            shaping = - (0.5 * holes + 0.1 * max_h)
            return game_points + survival_bonus + penalty + float(shaping)

        elif self.reward_type == "R3":
            # Potential-based reward shaping: R0 + (Phi(s') - Phi(s))
            current_potential = self._compute_potential(board_after)
            delta_potential = current_potential - self.prev_potential
            self.prev_potential = current_potential
            penalty = -20.0 if is_game_over else 0.0
            return game_points + delta_potential + penalty

        else:
            raise ValueError(f"reward_type '{self.reward_type}' tidak valid.")

    def _get_info(self, step_lines_cleared: int = 0, is_game_over: bool = False) -> Dict[str, Any]:
        """Mengumpulkan dictionary informasi status permainan."""
        holes = compute_holes(self.board)
        max_h = compute_max_height(self.board)
        bump = compute_bumpiness(self.board)
        agg_h = compute_aggregate_height(self.board)

        return {
            "action_mask": self.action_masks(),
            "score": int(self.score),
            "lines_cleared": int(self.lines_cleared),
            "step_lines_cleared": int(step_lines_cleared),
            "holes": int(holes),
            "max_height": int(max_h),
            "bumpiness": int(bump),
            "aggregate_height": int(agg_h),
            "steps": int(self.steps),
            "current_piece": int(self.current_piece),
            "next_piece": int(self.next_piece),
            "is_game_over": bool(is_game_over),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Mereset lingkungan ke kondisi awal episode baru."""
        super().reset(seed=seed)

        # Reset generator acak jika seed diberikan
        if seed is not None:
            self.piece_generator.reset(seed=seed)

        self.board = self.engine.create_board(self.board_height, self.board_width)
        self.current_piece = self.piece_generator.next_piece()
        self.next_piece = self.piece_generator.next_piece()

        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.prev_potential = self._compute_potential(self.board)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Mengeksekusi satu aksi penempatan piece pada lingkungan."""
        # 1. Validasi aksi terhadap action mask
        mask = self.action_masks()
        rot, col = decode_action(int(action), width=self.board_width)

        if not mask[action]:
            # Jika aksi ilegal dipilih: penalti berat dan terminasi episode
            is_game_over = True
            reward = -50.0
            terminated = True
            truncated = False
            info = self._get_info(step_lines_cleared=0, is_game_over=True)
            return self._get_obs(), float(reward), terminated, truncated, info

        # 2. Eksekusi penempatan piece (hard drop)
        board_before = self.board
        res: PlacementResult = self.engine.place_piece(
            board=self.board,
            piece_id=self.current_piece,
            rotation=rot,
            col=col,
        )

        self.board = res.board
        self.steps += 1

        if res.lines_cleared > 0:
            self.lines_cleared += res.lines_cleared
            self.score += LINE_CLEAR_POINTS.get(res.lines_cleared, 0)

        # 3. Hitung terminasi dan pemotongan langkah
        terminated = bool(res.is_game_over)
        truncated = bool((self.max_steps is not None) and (self.steps >= self.max_steps))

        # 4. Hitung reward pelatihan
        reward = self._compute_reward(
            lines_cleared=res.lines_cleared,
            is_game_over=res.is_game_over,
            board_before=board_before,
            board_after=self.board,
        )

        # 5. Gulirkan piece berikutnya
        if not terminated:
            self.current_piece = self.next_piece
            self.next_piece = self.piece_generator.next_piece()

        obs = self._get_obs()
        info = self._get_info(step_lines_cleared=res.lines_cleared, is_game_over=res.is_game_over)

        return obs, float(reward), terminated, truncated, info

    def valid_placements(self) -> List[ValidPlacement]:
        """Mengembalikan daftar seluruh afterstate yang sah untuk piece aktif saat ini."""
        return self.engine.get_valid_placements(self.board, self.current_piece)

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Merender tampilan lingkungan sesuai render_mode."""
        if self.render_mode is None:
            return None

        if self.render_mode == "ansi":
            return render_board_text(
                board=self.board,
                current_piece_id=self.current_piece,
                next_piece_id=self.next_piece,
                score=self.score,
                lines_cleared=self.lines_cleared,
                steps=self.steps,
                use_unicode=True,
            )

        if self.render_mode in ("rgb_array", "human"):
            if self._renderer is None:
                is_headless = (self.render_mode == "rgb_array")
                self._renderer = PygameRenderer(
                    board_height=self.board_height,
                    board_width=self.board_width,
                    cell_size=24,
                    headless=is_headless,
                )

            frame = self._renderer.render_frame(
                board=self.board,
                active_piece_id=self.current_piece,
                next_piece_id=self.next_piece,
                score=self.score,
                lines_cleared=self.lines_cleared,
                steps=self.steps,
            )
            return frame

        raise ValueError(f"render_mode '{self.render_mode}' tidak didukung.")

    def close(self) -> None:
        """Menutup sumber daya renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
