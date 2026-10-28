"""Protokol Evaluasi Resmi untuk Agen Tetris RL.

Sesuai Dokumen Proyek Bagian 3.3, 3.5, 5.3, dan 7:
- Evaluasi multi-seed independen (minimal 5 seed).
- Seed evaluasi strictly terpisah dari seed pelatihan.
- Batas maksimal episode evaluasi: 10.000 piece (dilaporkan sebagai data tersensor).
- Mencatat seluruh metrik: Skor game resmi, Baris terhapus, Langkah bertahan (steps),
  Tingkat Tetris (4 baris sekaligus), Rata-rata lubang, dan Tinggi tumpukan.
- Menghasilkan ringkasan agregat IQM, 95% Bootstrap CI, dan Kaplan-Meier survival data.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Union
import numpy as np

from tetris_rl.env import TetrisEnv
from tetris_rl.eval.stats import (
    EvalStats,
    compute_kaplan_meier_survival,
    summarize_metric,
)


class EpisodeResult(NamedTuple):
    """Hasil dari satu episode evaluasi."""
    seed: int
    score: int
    lines_cleared: int
    steps_survived: int
    tetris_count: int         # Berapa kali berhasil membersihkan 4 baris sekaligus
    total_clears_count: int   # Berapa kali aksi pembersihan baris terjadi
    avg_holes: float
    max_height: int
    is_censored: bool         # True jika menyentuh batas 10.000 piece


class BenchmarkSummary(NamedTuple):
    """Ringkasan komprehensif evaluasi multi-episode & multi-seed."""
    agent_name: str
    num_episodes: int
    score_stats: EvalStats
    lines_stats: EvalStats
    steps_stats: EvalStats
    tetris_rate: float        # Proporsi Tetris (4 baris) dari seluruh pembersihan
    avg_holes: float
    avg_max_height: float
    censored_ratio: float
    survival_timeline: np.ndarray
    survival_probabilities: np.ndarray
    episodes: List[EpisodeResult]


def evaluate_agent(
    agent: Any,
    env: TetrisEnv,
    seeds: List[int],
    episodes_per_seed: int = 5,
    max_steps_per_episode: int = 10000,
    agent_name: str = "Agent",
) -> BenchmarkSummary:
    """Menjalankan evaluasi resmi multi-seed untuk suatu agen.

    Args:
        agent: Objek agen yang memiliki metode act(obs, info) atau act(env).
        env: Instance TetrisEnv.
        seeds: Daftar seed evaluasi (harus berbeda dari seed pelatihan).
        episodes_per_seed: Jumlah episode per seed.
        max_steps_per_episode: Batas langkah (default 10.000 piece).
        agent_name: Nama agen untuk pelaporan.

    Returns:
        BenchmarkSummary: Ringkasan statistik lengkap hasil evaluasi.
    """
    # Pastikan max_steps env sesuai dengan batas evaluasi
    env.max_steps = max_steps_per_episode

    episode_results: List[EpisodeResult] = []

    for seed_idx, base_seed in enumerate(seeds):
        for ep in range(episodes_per_seed):
            ep_seed = base_seed * 1000 + ep
            obs, info = env.reset(seed=ep_seed)

            ep_steps = 0
            ep_tetris = 0
            ep_total_clears = 0
            holes_history: List[int] = []

            while True:
                # Adaptor pemanggilan act() untuk berbagai jenis agen
                if hasattr(agent, "evaluate_placement"):
                    # Agen heuristik berbasis afterstate
                    action = agent.act(env, obs=obs, info=info)
                elif hasattr(agent, "act"):
                    try:
                        action = agent.act(obs=obs, info=info)
                    except TypeError:
                        action = agent.act(env)
                elif hasattr(agent, "predict"):
                    # Agen Stable-Baselines3 / SB3
                    mask = info.get("action_mask")
                    if mask is not None:
                        action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
                    else:
                        action, _ = agent.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    raise ValueError(f"Agen tipe {type(agent)} tidak memiliki metode act atau predict.")

                obs, reward, terminated, truncated, info = env.step(action)
                ep_steps += 1
                holes_history.append(info["holes"])

                if info["step_lines_cleared"] > 0:
                    ep_total_clears += 1
                    if info["step_lines_cleared"] == 4:
                        ep_tetris += 1

                if terminated or truncated:
                    is_censored = bool(ep_steps >= max_steps_per_episode)
                    episode_results.append(
                        EpisodeResult(
                            seed=ep_seed,
                            score=int(info["score"]),
                            lines_cleared=int(info["lines_cleared"]),
                            steps_survived=int(ep_steps),
                            tetris_count=int(ep_tetris),
                            total_clears_count=int(ep_total_clears),
                            avg_holes=float(np.mean(holes_history)) if holes_history else 0.0,
                            max_height=int(info["max_height"]),
                            is_censored=is_censored,
                        )
                    )
                    break

    # Kumpulkan array metrik
    scores = np.array([ep.score for ep in episode_results], dtype=np.float64)
    lines = np.array([ep.lines_cleared for ep in episode_results], dtype=np.float64)
    steps = np.array([ep.steps_survived for ep in episode_results], dtype=np.float64)

    # Hitung ringkasan statistik
    score_stats = summarize_metric(scores, censored_limit=max_steps_per_episode)
    lines_stats = summarize_metric(lines, censored_limit=max_steps_per_episode)
    steps_stats = summarize_metric(steps, censored_limit=max_steps_per_episode)

    total_tetris = sum(ep.tetris_count for ep in episode_results)
    total_clears = sum(ep.total_clears_count for ep in episode_results)
    tetris_rate = float(total_tetris / total_clears) if total_clears > 0 else 0.0

    timeline, survival_prob = compute_kaplan_meier_survival(steps, censored_limit=max_steps_per_episode)

    return BenchmarkSummary(
        agent_name=agent_name,
        num_episodes=len(episode_results),
        score_stats=score_stats,
        lines_stats=lines_stats,
        steps_stats=steps_stats,
        tetris_rate=tetris_rate,
        avg_holes=float(np.mean([ep.avg_holes for ep in episode_results])),
        avg_max_height=float(np.mean([ep.max_height for ep in episode_results])),
        censored_ratio=steps_stats.censored_ratio,
        survival_timeline=timeline,
        survival_probabilities=survival_prob,
        episodes=episode_results,
    )

