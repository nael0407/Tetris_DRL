"""Agen Afterstate Value-Based: Linear TD(0) & Deep Q-Network (DQN).

Sesuai Dokumen Proyek Bagian 5.2 dan 5.4:
- Mengevaluasi nilai afterstate papan s' yang dihasilkan oleh setiap penempatan legal:
  a* = argmax_{p in valid_placements} [ r(s, a, s') + gamma * V(s') ]
- Mendukung dua varian representasi untuk desain faktorial 2x2:
  1. LinearTDAfterstateAgent: Nilai V(s') = w^T * phi(s') dilatih dengan semi-gradient TD(0).
  2. DQNAfterstateAgent: Jaringan saraf tiruan V(s'; theta) dengan Double DQN, Replay Buffer,
     dan Target Network pada papan mentah (raw board) atau fitur.
"""

from collections import deque
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tetris_rl.board import ValidPlacement
from tetris_rl.features import (
    compute_aggregate_height,
    compute_bumpiness,
    compute_holes,
    extract_feature_vector,
)


class LinearTDAfterstateAgent:
    """Agen afterstate linear berbasis Semi-Gradient TD(0).

    Mewakili sel [Gradient-based (TD) + Fitur linear] pada desain faktorial 2x2.
    V(s') = w0 * agg_h + w1 * lines + w2 * holes + w3 * bumpiness.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        initial_weights: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

        if initial_weights is not None:
            self.weights = np.asarray(initial_weights, dtype=np.float32).copy()
        else:
            # Default inisialisasi bobot standar Dellacherie
            self.weights = np.array([-0.510066, 0.760666, -0.356630, -0.184483], dtype=np.float32)

    def extract_features(self, placement: ValidPlacement) -> np.ndarray:
        """Mengekstrak 4 fitur afterstate [agg_height, lines_cleared, holes, bumpiness]."""
        board = placement.next_board
        holes = float(compute_holes(board))
        bump = float(compute_bumpiness(board))
        agg_h = float(compute_aggregate_height(board))
        lines = float(placement.lines_cleared)
        return np.array([agg_h, lines, holes, bump], dtype=np.float32)

    def predict_value(self, phi: np.ndarray) -> float:
        """Menghitung nilai V(s') = w^T * phi."""
        return float(np.dot(self.weights, phi))

    def evaluate_placement(self, placement: ValidPlacement) -> float:
        """Evaluasi nilai penempatan dengan penalti game over ekstrim."""
        if placement.is_game_over:
            return -1e9
        phi = self.extract_features(placement)
        return self.predict_value(phi)

    def act(
        self,
        env: Any = None,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Memilih aksi penempatan terbaik berdasarkan V(s')."""
        if env is None:
            raise ValueError("LinearTDAfterstateAgent membutuhkan argumen 'env' untuk evaluasi valid_placements.")

        placements: List[ValidPlacement] = env.valid_placements()
        if not placements:
            return 0

        # Eksplorasi epsilon-greedy
        if not deterministic and self.rng.random() < self.epsilon:
            legal_placements = [p for p in placements if not p.is_game_over]
            pool = legal_placements if legal_placements else placements
            idx = int(self.rng.integers(0, len(pool)))
            return pool[idx].action

        best_action = placements[0].action
        best_val = -1e12

        for p in placements:
            val = self.evaluate_placement(p)
            if val > best_val or (val == best_val and p.landing_row > placements[0].landing_row):
                best_val = val
                best_action = p.action

        return best_action

    def update(
        self,
        features_curr: np.ndarray,
        reward: float,
        features_next: Optional[np.ndarray],
        done: bool,
    ) -> float:
        """Pembaruan Semi-Gradient TD(0): w <- w + alpha * delta * phi."""
        v_curr = self.predict_value(features_curr)
        if done or features_next is None:
            target = reward
        else:
            v_next = self.predict_value(features_next)
            target = reward + self.gamma * v_next

        td_error = target - v_curr
        # Normalized gradient step & clipping untuk stabilitas numerik
        norm_sq = float(np.sum(features_curr ** 2)) + 1e-6
        step = (self.alpha / (1.0 + 0.001 * norm_sq)) * td_error * features_curr
        step = np.clip(step, -1.0, 1.0)
        self.weights += step

        # Proyeksi ke unit sphere (L2 normalization) seperti standar Bertsekas & Tsitsiklis (1996)
        w_norm = float(np.linalg.norm(self.weights))
        if w_norm > 0:
            self.weights /= w_norm

        return float(td_error)

    def save(self, path: str) -> None:
        """Menyimpan bobot linear ke file .npy."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        """Memuat bobot linear dari file .npy."""
        self.weights = np.load(path).astype(np.float32)


class AfterstateValueNet(nn.Module):
    """Jaringan saraf tiruan untuk memprediksi nilai scalar V(s') dari afterstate."""

    def __init__(self, input_dim: int = 200, hidden_dims: Sequence[int] = (128, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        curr_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h

        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass menghasilkan estimasi nilai scalar V(s')."""
        return self.network(x).squeeze(-1)


class ReplayBuffer:
    """Experience replay buffer berkinerja tinggi menggunakan NumPy array berukuran tetap."""

    def __init__(self, capacity: int, state_dim: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)

        self.idx = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Menyimpan satu transisi ke buffer."""
        self.states[self.idx] = state
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mengambil sampel acak berukuran batch_size dalam bentuk PyTorch Tensor."""
        indices = self.rng.integers(0, self.size, size=batch_size)
        states_b = torch.from_numpy(self.states[indices])
        rewards_b = torch.from_numpy(self.rewards[indices])
        next_states_b = torch.from_numpy(self.next_states[indices])
        dones_b = torch.from_numpy(self.dones[indices])
        return states_b, rewards_b, next_states_b, dones_b

    def __len__(self) -> int:
        return self.size


class DQNAfterstateAgent:
    """Deep Q-Network Afterstate Agent dengan Target Network dan Double DQN.

    Mewakili sel [Gradient-based (TD) + Papan mentah] atau [TD + Fitur] pada desain faktorial 2x2.
    """

    def __init__(
        self,
        board_height: int = 20,
        board_width: int = 10,
        representation: str = "raw",  # "raw" atau "features"
        hidden_dims: Sequence[int] = (128, 64),
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.board_height = board_height
        self.board_width = board_width
        self.representation = representation.lower()
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)

        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.rng = np.random.default_rng(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.representation == "raw":
            self.state_dim = board_height * board_width
        elif self.representation == "features":
            self.state_dim = 4  # agg_h, lines, holes, bumpiness
        else:
            raise ValueError(f"Representasi '{representation}' tidak dikenal. Pilih 'raw' atau 'features'.")

        self.policy_net = AfterstateValueNet(self.state_dim, hidden_dims).to(self.device)
        self.target_net = AfterstateValueNet(self.state_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity, self.state_dim, seed=seed)

        self.step_count = 0
        self.train_step_count = 0

    def state_to_tensor(self, board: np.ndarray, lines_cleared: int = 0) -> np.ndarray:
        """Mengonversi afterstate board ke vektor representasi (raw flat atau 4 fitur)."""
        if self.representation == "raw":
            return board.flatten().astype(np.float32)
        else:
            holes = float(compute_holes(board))
            bump = float(compute_bumpiness(board))
            agg_h = float(compute_aggregate_height(board))
            lines = float(lines_cleared)
            return np.array([agg_h, lines, holes, bump], dtype=np.float32)

    def evaluate_placement(self, placement: ValidPlacement) -> float:
        """Mengevaluasi satu penempatan dengan policy_net."""
        if placement.is_game_over:
            return -1e9

        feat = self.state_to_tensor(placement.next_board, placement.lines_cleared)
        with torch.no_grad():
            tensor = torch.from_numpy(feat).unsqueeze(0).to(self.device)
            val = float(self.policy_net(tensor).item())
        return val

    def act(
        self,
        env: Any = None,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Memilih aksi penempatan terbaik berdasarkan evaluasi afterstate."""
        if env is None:
            raise ValueError("DQNAfterstateAgent membutuhkan 'env' untuk mengakses valid_placements.")

        placements: List[ValidPlacement] = env.valid_placements()
        if not placements:
            return 0

        # Eksplorasi epsilon-greedy
        if not deterministic and self.rng.random() < self.epsilon:
            legal = [p for p in placements if not p.is_game_over]
            pool = legal if legal else placements
            idx = int(self.rng.integers(0, len(pool)))
            return int(pool[idx].action)

        # Batch forward pass untuk seluruh penempatan legal agar lebih cepat
        feats = np.empty((len(placements), self.state_dim), dtype=np.float32)
        for i, p in enumerate(placements):
            feats[i] = self.state_to_tensor(p.next_board, p.lines_cleared)

        with torch.no_grad():
            tensor_b = torch.from_numpy(feats).to(self.device)
            values = self.policy_net(tensor_b).cpu().numpy()

        # Beri penalti ekstrim pada penempatan game over
        for i, p in enumerate(placements):
            if p.is_game_over:
                values[i] = -1e9

        best_idx = int(np.argmax(values))
        return int(placements[best_idx].action)

    def decay_epsilon(self) -> None:
        """Memperbarui nilai epsilon sesuai jadwal peluruhan linear."""
        self.step_count += 1
        fraction = min(1.0, float(self.step_count) / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def remember(
        self,
        board_after: np.ndarray,
        lines_cleared: int,
        reward: float,
        next_board_after: Optional[np.ndarray],
        next_lines_cleared: int,
        done: bool,
    ) -> None:
        """Menyimpan transisi afterstate ke replay buffer."""
        curr_feat = self.state_to_tensor(board_after, lines_cleared)
        if next_board_after is not None:
            next_feat = self.state_to_tensor(next_board_after, next_lines_cleared)
        else:
            next_feat = np.zeros(self.state_dim, dtype=np.float32)

        self.buffer.push(curr_feat, reward, next_feat, done)

    def update(self) -> Optional[float]:
        """Melakukan satu langkah optimasi gradien PyTorch jika buffer mencukupi."""
        if len(self.buffer) < self.batch_size:
            return None

        states, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Double DQN: pemilihan nilai target oleh target_net
        with torch.no_grad():
            next_values = self.target_net(next_states)
            target = rewards + (1.0 - dones.float()) * self.gamma * next_values

        curr_values = self.policy_net(states)
        loss = self.criterion(curr_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        """Menyimpan checkpoint model PyTorch."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "policy_net_state": self.policy_net.state_dict(),
                "target_net_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "train_step_count": self.train_step_count,
                "epsilon": self.epsilon,
                "representation": self.representation,
                "state_dim": self.state_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Memuat checkpoint model PyTorch."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step_count = checkpoint.get("step_count", 0)
        self.train_step_count = checkpoint.get("train_step_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
