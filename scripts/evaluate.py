"""Skrip resmi eksekusi protokol evaluasi agen (evaluate.py).

Menjalankan evaluasi multi-seed untuk konfigurasi tertentu, menghitung metrik IQM,
bootstrap CI 95%, rasio data tersensor, dan kurva survival Kaplan-Meier.
Hasil ringkasan disimpan ke results/summary/ dalam format JSON dan CSV.

Penggunaan:
    python scripts/evaluate.py --config configs/baseline_random.yaml
"""

import argparse
import json
import os
from typing import Any, Dict
import numpy as np
import yaml

from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.eval.evaluate import BenchmarkSummary, evaluate_agent


def load_config(config_path: str) -> Dict[str, Any]:
    """Membaca konfigurasi eksperimen dari file YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(config: Dict[str, Any]) -> Any:
    """Instansiasi agen berdasarkan file konfigurasi."""
    agent_type = config.get("agent_type", "random").lower()
    if agent_type == "random":
        return RandomAgent()
    elif agent_type == "heuristic":
        agent_cfg = config.get("agent", {})
        weights = agent_cfg.get("weights")
        name = agent_cfg.get("name", "Heuristic")
        return HeuristicAgent(weights=weights, name=name)
    else:
        raise ValueError(f"Agent type '{agent_type}' belum didukung di evaluate.py.")


def run_evaluation(config_path: str) -> BenchmarkSummary:
    """Mengeksekusi evaluasi agen dan menyimpan hasilnya."""
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})

    env = TetrisEnv(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        obs_type=env_cfg.get("obs_type", "raw"),
        reward_type=env_cfg.get("reward_type", "R0"),
        piece_generator_type=env_cfg.get("piece_generator", "uniform"),
        use_numba=env_cfg.get("use_numba", True),
    )

    agent = build_agent(cfg)
    seeds = eval_cfg.get("seeds", [101, 102, 103, 104, 105])
    eps_per_seed = eval_cfg.get("episodes_per_seed", 5)
    max_steps = eval_cfg.get("max_steps_per_episode", 10000)
    agent_name = cfg.get("experiment_id", "Agent")

    print(f"\n[Evaluasi] Memulai evaluasi: {agent_name}")
    print(f"[Evaluasi] Seeds: {seeds} ({eps_per_seed} ep/seed = total {len(seeds) * eps_per_seed} ep)")
    print(f"[Evaluasi] Batas episode: {max_steps:,} piece")

    summary = evaluate_agent(
        agent=agent,
        env=env,
        seeds=seeds,
        episodes_per_seed=eps_per_seed,
        max_steps_per_episode=max_steps,
        agent_name=agent_name,
    )

    # Cetak tabel ringkasan di terminal
    print("\n" + "=" * 65)
    print(f" RINGKASAN EVALUASI: {summary.agent_name.upper()}")
    print("=" * 65)
    print(f" Total Episode      : {summary.num_episodes}")
    print(f" Skor (IQM [95% CI]): {summary.score_stats.iqm:.1f} [{summary.score_stats.ci_low:.1f}, {summary.score_stats.ci_high:.1f}] (Mean: {summary.score_stats.mean:.1f})")
    print(f" Baris (IQM [CI])   : {summary.lines_stats.iqm:.1f} [{summary.lines_stats.ci_low:.1f}, {summary.lines_stats.ci_high:.1f}] (Mean: {summary.lines_stats.mean:.1f})")
    print(f" Langkah Bertahan   : {summary.steps_stats.iqm:.1f} (Mean: {summary.steps_stats.mean:.1f})")
    print(f" Rasio Tersensor    : {summary.censored_ratio * 100:.1f}% (mencapai batas {max_steps} piece)")
    print(f" Tingkat Tetris (4x): {summary.tetris_rate * 100:.2f}%")
    print(f" Rata-rata Lubang   : {summary.avg_holes:.2f}")
    print("=" * 65)

    # Simpan hasil ringkasan ke results/summary/
    out_dir = os.path.join("results", "summary")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{summary.agent_name}.json")

    json_data = {
        "agent_name": summary.agent_name,
        "num_episodes": summary.num_episodes,
        "score": {
            "iqm": summary.score_stats.iqm,
            "ci_low": summary.score_stats.ci_low,
            "ci_high": summary.score_stats.ci_high,
            "mean": summary.score_stats.mean,
            "std": summary.score_stats.std,
            "median": summary.score_stats.median,
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
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"[Simpan] Hasil ringkasan tersimpan di: {out_json}\n")
    env.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Agen Tetris RL")
    parser.add_argument("--config", type=str, required=True, help="Path ke file konfigurasi YAML")
    args = parser.parse_args()

    run_evaluation(args.config)

