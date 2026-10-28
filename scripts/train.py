"""Skrip resmi pelatihan agen (train.py).

Menjalankan proses pelatihan sesuai file konfigurasi eksperimen.
Mendukung pelatihan Cross-Entropy Method (CEM) dan mencatat akuntansi langkah interaksi
secara jujur dan transparan.

Penggunaan:
    python scripts/train.py --config configs/e1_cem_features.yaml
"""

import argparse
import json
import os
from typing import Any, Dict
import numpy as np
import yaml

from tetris_rl.agents.cem import CEMAgent, CEMResult
from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.eval.evaluate import evaluate_agent


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_cem(config: Dict[str, Any]) -> CEMResult:
    """Melatih agen CEM pada representasi fitur linear."""
    env_cfg = config.get("env", {})
    cem_cfg = config.get("cem", {})

    env = TetrisEnv(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        obs_type=env_cfg.get("obs_type", "features"),
        reward_type=env_cfg.get("reward_type", "R0"),
        piece_generator_type=env_cfg.get("piece_generator", "uniform"),
        use_numba=env_cfg.get("use_numba", True),
    )

    cem = CEMAgent(
        num_features=cem_cfg.get("num_features", 4),
        population_size=cem_cfg.get("population_size", 20),
        elite_ratio=cem_cfg.get("elite_ratio", 0.20),
        initial_std=cem_cfg.get("initial_std", 0.5),
        noise_std=cem_cfg.get("noise_std", 0.05),
        noise_decay=cem_cfg.get("noise_decay", 0.98),
        seed=cem_cfg.get("seed", 42),
    )

    print("\n" + "=" * 65)
    print(f" MEMULAI PELATIHAN CEM: {config.get('experiment_id')}")
    print("=" * 65)
    print(f" Populasi      : {cem.population_size} | Elit: {cem.num_elites}")
    print(f" Maks Iterasi  : {cem_cfg.get('max_iterations', 10)}")
    print(f" Game / Kandidat: {cem_cfg.get('episodes_per_candidate', 2)}")
    print("=" * 65 + "\n")

    def progress_callback(iteration: int, elite_score: float, total_pieces: int) -> None:
        print(f"[Iterasi {iteration:02d}] Rata-rata Skor Elit: {elite_score:8.1f} | Total Interaksi Piece: {total_pieces:,}")

    res = cem.train(
        env=env,
        max_iterations=cem_cfg.get("max_iterations", 10),
        episodes_per_candidate=cem_cfg.get("episodes_per_candidate", 2),
        callback=progress_callback,
    )

    print("\n" + "=" * 65)
    print(" PELATIHAN CEM SELESAI")
    print("=" * 65)
    print(f" Bobot Terbaik Ditemukan : {np.round(res.best_weights, 5).tolist()}")
    print(f" Skor Terbaik Kandidat   : {res.best_score:.1f}")
    print(f" Total Piece Dimainkan   : {res.total_interaction_pieces:,} piece")
    print("=" * 65 + "\n")

    # Simpan model bobot akhir ke models/final/
    os.makedirs(os.path.join("models", "final"), exist_ok=True)
    model_path = os.path.join("models", "final", "cem_features_best.npy")
    np.save(model_path, res.best_weights)
    print(f"[Simpan Model] Bobot terbaik disimpan di: {model_path}")

    # Simpan riwayat pelatihan ke results/summary/
    os.makedirs(os.path.join("results", "summary"), exist_ok=True)
    summary_path = os.path.join("results", "summary", f"{config.get('experiment_id')}_train.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": config.get("experiment_id"),
                "best_weights": res.best_weights.tolist(),
                "best_score": res.best_score,
                "total_iterations": res.total_iterations,
                "total_interaction_pieces": res.total_interaction_pieces,
                "history_scores": res.history_scores,
            },
            f,
            indent=2,
        )
    print(f"[Simpan Log] Riwayat pelatihan disimpan di: {summary_path}")

    # Jalankan evaluasi resmi multi-seed terhadap model terbaik
    eval_cfg = config.get("eval", {})
    eval_seeds = eval_cfg.get("seeds", [3001, 3002, 3003, 3004, 3005])
    eps_per_seed = eval_cfg.get("episodes_per_seed", 5)
    max_steps = eval_cfg.get("max_steps_per_episode", 10000)

    print("\n[Evaluasi Resmi] Mengevaluasi model terbaik pada 5 seed evaluasi independen...")
    best_agent = HeuristicAgent(weights=res.best_weights, name="CEM-Trained-Agent")
    summary = evaluate_agent(
        agent=best_agent,
        env=env,
        seeds=eval_seeds,
        episodes_per_seed=eps_per_seed,
        max_steps_per_episode=max_steps,
        agent_name=config.get("experiment_id", "e1_cem_features"),
    )

    eval_out = os.path.join("results", "summary", f"{config.get('experiment_id')}.json")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent_name": summary.agent_name,
                "num_episodes": summary.num_episodes,
                "total_interaction_pieces": res.total_interaction_pieces,
                "score": {
                    "iqm": summary.score_stats.iqm,
                    "ci_low": summary.score_stats.ci_low,
                    "ci_high": summary.score_stats.ci_high,
                    "mean": summary.score_stats.mean,
                },
                "lines_cleared": {
                    "iqm": summary.lines_stats.iqm,
                    "ci_low": summary.lines_stats.ci_low,
                    "ci_high": summary.lines_stats.ci_high,
                    "mean": summary.lines_stats.mean,
                },
                "steps_survived": {
                    "iqm": summary.steps_stats.iqm,
                    "mean": summary.steps_stats.mean,
                    "censored_ratio": summary.censored_ratio,
                },
                "tetris_rate": summary.tetris_rate,
                "avg_holes": summary.avg_holes,
                "avg_max_height": summary.avg_max_height,
                "survival_timeline": summary.survival_timeline.tolist(),
                "survival_probabilities": summary.survival_probabilities.tolist(),
            },
            f,
            indent=2,
        )

    print(f"\n[Selesai] Skor Evaluasi IQM: {summary.score_stats.iqm:.1f} [{summary.score_stats.ci_low:.1f}, {summary.score_stats.ci_high:.1f}]")
    print(f"[Selesai] Baris Dihapus IQM: {summary.lines_stats.iqm:.1f}")
    print(f"[Simpan] Ringkasan evaluasi tersimpan di: {eval_out}\n")

    env.close()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Agen Tetris RL")
    parser.add_argument("--config", type=str, required=True, help="Path ke konfigurasi YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_type = cfg.get("agent_type", "").lower()
    if agent_type == "cem":
        train_cem(cfg)
    else:
        raise ValueError(f"Agent type '{agent_type}' belum didukung di train.py.")

