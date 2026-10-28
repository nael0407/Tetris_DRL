"""Agen Cross-Entropy Method (CEM) untuk Optimasi Bobot Fitur Linear.

Sesuai Dokumen Proyek Bagian 5.1, 5.3, dan 5.4:
- Mengoptimasi parameter bobot w dalam fungsi nilai linear afterstate secara gradient-free.
- Menjaga distribusi Gaussian N(mu, diag(sigma^2)) pada ruang parameter bobot.
- Menghitung langkah interaksi secara ketat dan jujur:
  total_interaction_steps = total piece yang dimainkan dari seluruh evaluasi kandidat
  (bukan sekadar menghitung generasi).
"""

import os
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from tetris_rl.agents.heuristic import DEFAULT_HEURISTIC_WEIGHTS, HeuristicAgent
from tetris_rl.env import TetrisEnv


class CEMResult(NamedTuple):
    """Hasil proses pelatihan CEM."""
    best_weights: np.ndarray
    best_score: float
    total_iterations: int
    total_interaction_pieces: int
    history_scores: List[float]


class CEMAgent:
    """Implementasi algoritma Cross-Entropy Method (CEM) untuk Tetris."""

    def __init__(
        self,
        num_features: int = 4,
        population_size: int = 20,
        elite_ratio: float = 0.20,
        initial_mean: Optional[np.ndarray] = None,
        initial_std: float = 0.5,
        noise_std: float = 0.05,
        noise_decay: float = 0.98,
        seed: int = 42,
    ) -> None:
        """Inisialisasi pengoptimal CEM.

        Args:
            num_features: Dimensi parameter bobot yang dioptimasi (default 4).
            population_size: Jumlah kandidat parameter per iterasi (populasi).
            elite_ratio: Proporsi kandidat teratas yang dijadikan elit (misal 0.2 = 20%).
            initial_mean: Rata-rata awal distribusi Gaussian (default: bobot literatur).
            initial_std: Simpangan baku awal distribusi parameter.
            noise_std: Noise tambahan pada kovariansi untuk mencegah konvergensi dini.
            noise_decay: Faktor peluruhan noise per generasi.
            seed: Seed bilangan acak untuk reproduktibilitas.
        """
        self.num_features = num_features
        self.population_size = population_size
        self.num_elites = max(1, int(population_size * elite_ratio))
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.rng = np.random.default_rng(seed)

        if initial_mean is not None:
            self.mean = np.asarray(initial_mean, dtype=np.float32).copy()
        else:
            self.mean = DEFAULT_HEURISTIC_WEIGHTS.copy()

        self.std = np.full(self.num_features, initial_std, dtype=np.float32)
        self.total_interaction_pieces: int = 0
        self.best_weights = self.mean.copy()
        self.best_score = -1e9

    def _sample_population(self) -> np.ndarray:
        """Membangkitkan kandidat bobot dari distribusi Gaussian dan menormalisasi panjangnya."""
        # Sampel bentuk (P, num_features)
        raw_samples = self.rng.normal(
            loc=self.mean,
            scale=self.std,
            size=(self.population_size, self.num_features),
        )
        # Normalisasi L2 setiap vektor bobot agar skalanya konsisten
        norms = np.linalg.norm(raw_samples, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return raw_samples / norms

    def evaluate_candidate(
        self,
        weights: np.ndarray,
        env: TetrisEnv,
        num_episodes: int = 3,
        eval_seed_start: int = 1000,
    ) -> float:
        """Mengevaluasi satu set bobot kandidat selama beberapa episode game.

        Mencatat setiap langkah piece yang dimainkan ke total_interaction_pieces.
        """
        agent = HeuristicAgent(weights=weights)
        total_score = 0.0

        for ep in range(num_episodes):
            ep_seed = eval_seed_start + ep
            obs, info = env.reset(seed=ep_seed)
            ep_score = 0.0

            while True:
                action = agent.act(env, obs=obs, info=info)
                obs, reward, terminated, truncated, info = env.step(action)
                self.total_interaction_pieces += 1  # Akuntansi sampel jujur per piece

                if terminated or truncated:
                    ep_score = float(info["score"])
                    break

            total_score += ep_score

        return float(total_score / num_episodes)

    def train(
        self,
        env: TetrisEnv,
        max_iterations: int = 10,
        episodes_per_candidate: int = 2,
        target_pieces_budget: Optional[int] = None,
        callback: Optional[Callable[[int, float, int], None]] = None,
    ) -> CEMResult:
        """Menjalankan loop optimasi Cross-Entropy Method.

        Args:
            env: Lingkungan game Tetris.
            max_iterations: Maksimal jumlah generasi pelatihan.
            episodes_per_candidate: Jumlah game evaluasi untuk tiap kandidat.
            target_pieces_budget: Jika diatur, berhenti saat total piece melebihi anggaran ini.
            callback: Fungsi callback opsional (iterasi, skor_rata_rata_elit, total_pieces).

        Returns:
            CEMResult: Hasil bobot terbaik, skor, dan riwayat pelatihan.
        """
        history_scores: List[float] = []

        for iteration in range(max_iterations):
            if target_pieces_budget is not None and self.total_interaction_pieces >= target_pieces_budget:
                break

            # 1. Sampel populasi
            candidates = self._sample_population()
            scores = np.empty(self.population_size, dtype=np.float64)

            # 2. Evaluasi seluruh kandidat
            base_seed = int(self.rng.integers(1, 1_000_000))
            for i in range(self.population_size):
                scores[i] = self.evaluate_candidate(
                    weights=candidates[i],
                    env=env,
                    num_episodes=episodes_per_candidate,
                    eval_seed_start=base_seed,
                )

            # 3. Seleksi Elit (Kandidat dengan skor tertinggi)
            elite_indices = np.argsort(scores)[::-1][: self.num_elites]
            elites = candidates[elite_indices]
            elite_scores = scores[elite_indices]

            mean_elite_score = float(np.mean(elite_scores))
            history_scores.append(mean_elite_score)

            # Perbarui kandidat terbaik global
            if elite_scores[0] > self.best_score:
                self.best_score = float(elite_scores[0])
                self.best_weights = elites[0].copy()

            # 4. Perbarui distribusi Gaussian (Mean & Simpangan Baku Elit)
            self.mean = np.mean(elites, axis=0)
            # Normalisasi mean
            mean_norm = np.linalg.norm(self.mean)
            if mean_norm > 0:
                self.mean /= mean_norm

            raw_std = np.std(elites, axis=0)
            # Tambahkan noise eksplorasi
            self.std = raw_std + self.noise_std
            self.noise_std = max(0.005, self.noise_std * self.noise_decay)

            if callback is not None:
                callback(iteration + 1, mean_elite_score, self.total_interaction_pieces)

        return CEMResult(
            best_weights=self.best_weights,
            best_score=self.best_score,
            total_iterations=len(history_scores),
            total_interaction_pieces=self.total_interaction_pieces,
            history_scores=history_scores,
        )

