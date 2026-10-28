"""Skrip resmi pelatihan agen (train.py).

Menjalankan proses pelatihan untuk seluruh agen Tetris RL:
1. CEM (Cross-Entropy Method pada fitur linear)
2. Linear TD(0) Afterstate (Semi-Gradient TD pada fitur linear)
3. DQN Afterstate (Deep Q-Network pada afterstate papan mentah atau fitur)
4. MaskablePPO (Policy Gradient dengan action masking)

Mendukung logging ke TensorBoard, penyimpanan checkpoint teratur,
akuntansi piece interaksi yang jujur, dan protokol evaluasi multi-seed otomatis.

Penggunaan:
    python scripts/train.py --config configs/e1_cem_features.yaml
    python scripts/train.py --config configs/e2_td_features.yaml
    python scripts/train.py --config configs/e2_td_net.yaml
    python scripts/train.py --config configs/e3_ppo_masked.yaml
"""

import argparse
import json
import os
import time
from typing import Any, Dict
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from tetris_rl.agents.cem import CEMAgent, CEMResult
from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.agents.ppo_masked import (
    MaskedPPOAgent,
    make_masked_tetris_env,
    tetris_mask_fn,
)
from tetris_rl.agents.td_afterstate import (
    DQNAfterstateAgent,
    LinearTDAfterstateAgent,
)
from tetris_rl.env import TetrisEnv
from tetris_rl.eval.evaluate import evaluate_agent


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_eval_summary(summary, output_path: str, total_interaction_pieces: int) -> None:
    """Menyimpan ringkasan metrik evaluasi resmi ke format JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent_name": summary.agent_name,
                "num_episodes": summary.num_episodes,
                "total_interaction_pieces": total_interaction_pieces,
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
    print(f"[Simpan] Ringkasan evaluasi tersimpan di: {output_path}")


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

    def progress_callback(iteration: int, elite_score: float, total_pieces: int) -> None:
        print(f"[Iterasi {iteration:02d}] Rata-rata Skor Elit: {elite_score:8.1f} | Total Interaksi Piece: {total_pieces:,}")

    res = cem.train(
        env=env,
        max_iterations=cem_cfg.get("max_iterations", 10),
        episodes_per_candidate=cem_cfg.get("episodes_per_candidate", 2),
        callback=progress_callback,
    )

    # Simpan bobot terbaik
    os.makedirs(os.path.join("models", "final"), exist_ok=True)
    model_path = os.path.join("models", "final", f"{config.get('experiment_id')}_best.npy")
    np.save(model_path, res.best_weights)
    print(f"[Simpan Model] Bobot terbaik disimpan di: {model_path}")

    # Evaluasi multi-seed
    eval_cfg = config.get("eval", {})
    best_agent = HeuristicAgent(weights=res.best_weights, name="CEM-Trained-Agent")
    summary = evaluate_agent(
        agent=best_agent,
        env=env,
        seeds=eval_cfg.get("seeds", [3001, 3002, 3003, 3004, 3005]),
        episodes_per_seed=eval_cfg.get("episodes_per_seed", 5),
        max_steps_per_episode=eval_cfg.get("max_steps_per_episode", 10000),
        agent_name=config.get("experiment_id", "cem"),
    )

    eval_out = os.path.join("results", "summary", f"{config.get('experiment_id')}.json")
    save_eval_summary(summary, eval_out, res.total_interaction_pieces)
    env.close()
    return res


def train_td_linear(config: Dict[str, Any]) -> LinearTDAfterstateAgent:
    """Melatih agen Linear TD(0) Afterstate."""
    env_cfg = config.get("env", {})
    td_cfg = config.get("td", {})

    env = TetrisEnv(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        obs_type=env_cfg.get("obs_type", "features"),
        reward_type=env_cfg.get("reward_type", "R3"),
        use_numba=env_cfg.get("use_numba", True),
    )

    agent = LinearTDAfterstateAgent(
        alpha=td_cfg.get("alpha", 0.001),
        gamma=td_cfg.get("gamma", 0.99),
        epsilon=td_cfg.get("epsilon", 0.05),
        seed=td_cfg.get("seed", 42),
    )

    total_timesteps = int(td_cfg.get("total_timesteps", 50000))
    print(f"\n[Pelatihan] Memulai Linear TD(0) ({total_timesteps:,} langkah)...")

    obs, info = env.reset(seed=td_cfg.get("seed", 42))
    total_pieces = 0
    episodes = 0
    ep_score = 0
    ep_lines = 0

    log_dir = os.path.join("results", "tensorboard", config.get("experiment_id", "td_linear"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    while total_pieces < total_timesteps:
        placements = env.valid_placements()
        if not placements:
            obs, info = env.reset()
            continue

        action = agent.act(env=env, deterministic=False)
        # Cari placement terpilih
        chosen_p = next((p for p in placements if p.action == action), placements[0])
        phi_curr = agent.extract_features(chosen_p)

        obs, reward, terminated, truncated, info = env.step(action)
        total_pieces += 1
        ep_score += info.get("score", 0)
        ep_lines += info.get("step_lines_cleared", 0)

        done = terminated or truncated
        phi_next = None

        if not done:
            next_placements = env.valid_placements()
            if next_placements:
                next_action = agent.act(env=env, deterministic=True)
                next_p = next((p for p in next_placements if p.action == next_action), next_placements[0])
                phi_next = agent.extract_features(next_p)

        td_err = agent.update(phi_curr, reward, phi_next, done)

        if total_pieces % 2000 == 0:
            writer.add_scalar("train/td_error", abs(td_err), total_pieces)
            writer.add_scalar("train/episodes", episodes, total_pieces)
            print(f"[{total_pieces:6,d}/{total_timesteps:,d} piece] TD Error: {td_err:7.3f} | Bobot: {np.round(agent.weights, 4).tolist()}")

        if done:
            writer.add_scalar("train/episode_score", info["score"], total_pieces)
            writer.add_scalar("train/episode_lines", info["lines_cleared"], total_pieces)
            episodes += 1
            obs, info = env.reset()

    writer.close()

    # Simpan bobot akhir
    model_path = os.path.join("models", "final", f"{config.get('experiment_id')}_best.npy")
    agent.save(model_path)
    print(f"\n[Simpan Model] Bobot Linear TD tersimpan di: {model_path}")

    # Evaluasi multi-seed
    eval_cfg = config.get("eval", {})
    summary = evaluate_agent(
        agent=agent,
        env=env,
        seeds=eval_cfg.get("seeds", [3001, 3002, 3003, 3004, 3005]),
        episodes_per_seed=eval_cfg.get("episodes_per_seed", 5),
        max_steps_per_episode=eval_cfg.get("max_steps_per_episode", 10000),
        agent_name=config.get("experiment_id", "td_linear"),
    )
    eval_out = os.path.join("results", "summary", f"{config.get('experiment_id')}.json")
    save_eval_summary(summary, eval_out, total_pieces)
    env.close()
    return agent


def train_dqn_afterstate(config: Dict[str, Any]) -> DQNAfterstateAgent:
    """Melatih agen Deep Q-Network Afterstate."""
    env_cfg = config.get("env", {})
    dqn_cfg = config.get("dqn", {})

    env = TetrisEnv(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        obs_type=env_cfg.get("obs_type", "raw"),
        reward_type=env_cfg.get("reward_type", "R3"),
        use_numba=env_cfg.get("use_numba", True),
    )

    agent = DQNAfterstateAgent(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        representation=dqn_cfg.get("representation", "raw"),
        hidden_dims=dqn_cfg.get("hidden_dims", (128, 64)),
        lr=float(dqn_cfg.get("lr", 5e-4)),
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        buffer_capacity=int(dqn_cfg.get("buffer_capacity", 50000)),
        batch_size=int(dqn_cfg.get("batch_size", 64)),
        target_update_freq=int(dqn_cfg.get("target_update_freq", 500)),
        epsilon_start=float(dqn_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(dqn_cfg.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(dqn_cfg.get("epsilon_decay_steps", 20000)),
        seed=dqn_cfg.get("seed", 42),
    )

    total_timesteps = int(dqn_cfg.get("total_timesteps", 50000))
    ckpt_interval = int(dqn_cfg.get("checkpoint_interval", 10000))
    exp_id = config.get("experiment_id", "dqn_afterstate")

    log_dir = os.path.join("results", "tensorboard", exp_id)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"\n[Pelatihan] Memulai DQN Afterstate ({total_timesteps:,} timesteps) di device {agent.device}...")

    obs, info = env.reset(seed=dqn_cfg.get("seed", 42))
    total_pieces = 0
    episodes = 0

    while total_pieces < total_timesteps:
        placements = env.valid_placements()
        if not placements:
            obs, info = env.reset()
            continue

        action = agent.act(env=env, deterministic=False)
        chosen_p = next((p for p in placements if p.action == action), placements[0])
        board_after = chosen_p.next_board.copy()
        lines_cleared = chosen_p.lines_cleared

        obs, reward, terminated, truncated, info = env.step(action)
        total_pieces += 1
        agent.decay_epsilon()

        done = terminated or truncated
        next_board_after = None
        next_lines_cleared = 0

        if not done:
            next_placements = env.valid_placements()
            if next_placements:
                next_act = agent.act(env=env, deterministic=True)
                next_p = next((p for p in next_placements if p.action == next_act), next_placements[0])
                next_board_after = next_p.next_board
                next_lines_cleared = next_p.lines_cleared

        agent.remember(board_after, lines_cleared, reward, next_board_after, next_lines_cleared, done)
        loss = agent.update()

        if loss is not None and total_pieces % 500 == 0:
            writer.add_scalar("train/loss", loss, total_pieces)
            writer.add_scalar("train/epsilon", agent.epsilon, total_pieces)

        if total_pieces % 2000 == 0:
            loss_str = f"{loss:.4f}" if loss is not None else "Buffer filling"
            print(f"[{total_pieces:6,d}/{total_timesteps:,d}] Epsilon: {agent.epsilon:5.3f} | Loss: {loss_str} | Replay Buffer: {len(agent.buffer):,}")

        if total_pieces % ckpt_interval == 0:
            ckpt_path = os.path.join("models", "checkpoints", f"{exp_id}_step_{total_pieces}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] Tersimpan di: {ckpt_path}")

        if done:
            episodes += 1
            writer.add_scalar("train/episode_score", info["score"], total_pieces)
            writer.add_scalar("train/episode_lines", info["lines_cleared"], total_pieces)
            writer.add_scalar("train/steps_survived", info["steps"], total_pieces)
            obs, info = env.reset()

    writer.close()

    # Simpan checkpoint terakhir
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join("models", "checkpoints", f"{exp_id}_step_{total_pieces}.pt")
    agent.save(ckpt_path)
    print(f"[Checkpoint] Checkpoint langkah {total_pieces} disimpan di: {ckpt_path}")

    # Simpan model terbaik
    final_path = os.path.join("models", "final", f"{exp_id}_best.pt")
    agent.save(final_path)
    print(f"\n[Simpan Model] Model DQN Afterstate final disimpan di: {final_path}")

    # Evaluasi multi-seed
    eval_cfg = config.get("eval", {})
    summary = evaluate_agent(
        agent=agent,
        env=env,
        seeds=eval_cfg.get("seeds", [3001, 3002, 3003, 3004, 3005]),
        episodes_per_seed=eval_cfg.get("episodes_per_seed", 5),
        max_steps_per_episode=eval_cfg.get("max_steps_per_episode", 10000),
        agent_name=exp_id,
    )
    eval_out = os.path.join("results", "summary", f"{exp_id}.json")
    save_eval_summary(summary, eval_out, total_pieces)
    env.close()
    return agent


def train_ppo_masked(config: Dict[str, Any]) -> MaskedPPOAgent:
    """Melatih agen MaskablePPO dengan Action Masking."""
    from sb3_contrib import MaskablePPO

    env_cfg = config.get("env", {})
    ppo_cfg = config.get("ppo", {})
    exp_id = config.get("experiment_id", "ppo_masked")

    train_env = make_masked_tetris_env(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        obs_type=env_cfg.get("obs_type", "raw"),
        reward_type=env_cfg.get("reward_type", "R3"),
        seed=ppo_cfg.get("seed", 42),
    )

    tb_log = os.path.join("results", "tensorboard", exp_id)
    os.makedirs(tb_log, exist_ok=True)

    total_timesteps = int(ppo_cfg.get("total_timesteps", 50000))
    print(f"\n[Pelatihan] Memulai MaskablePPO ({total_timesteps:,} timesteps)...")

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 512)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 5)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        verbose=1,
        tensorboard_log=tb_log,
        seed=ppo_cfg.get("seed", 42),
    )

    model.learn(total_timesteps=total_timesteps)

    # Simpan model final
    os.makedirs(os.path.join("models", "final"), exist_ok=True)
    final_path = os.path.join("models", "final", f"{exp_id}_best.zip")
    model.save(final_path)
    print(f"\n[Simpan Model] Model MaskablePPO final disimpan di: {final_path}")

    # Evaluasi multi-seed
    eval_cfg = config.get("eval", {})
    eval_env = TetrisEnv(
        board_height=env_cfg.get("board_height", 20),
        board_width=env_cfg.get("board_width", 10),
        obs_type=env_cfg.get("obs_type", "raw"),
        reward_type=env_cfg.get("reward_type", "R3"),
        use_numba=True,
    )

    agent = MaskedPPOAgent(model=model)
    summary = evaluate_agent(
        agent=agent,
        env=eval_env,
        seeds=eval_cfg.get("seeds", [3001, 3002, 3003, 3004, 3005]),
        episodes_per_seed=eval_cfg.get("episodes_per_seed", 5),
        max_steps_per_episode=eval_cfg.get("max_steps_per_episode", 10000),
        agent_name=exp_id,
    )

    eval_out = os.path.join("results", "summary", f"{exp_id}.json")
    save_eval_summary(summary, eval_out, total_timesteps)

    train_env.close()
    eval_env.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Agen Tetris RL")
    parser.add_argument("--config", type=str, required=True, help="Path ke konfigurasi YAML")
    parser.add_argument("--timesteps", type=int, default=None, help="Override jumlah timesteps pelatihan jika dispesifikasikan")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.timesteps is not None:
        if "td" in cfg:
            cfg["td"]["total_timesteps"] = args.timesteps
        if "dqn" in cfg:
            cfg["dqn"]["total_timesteps"] = args.timesteps
        if "ppo" in cfg:
            cfg["ppo"]["total_timesteps"] = args.timesteps

    agent_type = cfg.get("agent_type", "").lower()

    if agent_type == "cem":
        train_cem(cfg)
    elif agent_type == "td_linear":
        train_td_linear(cfg)
    elif agent_type == "dqn_afterstate":
        train_dqn_afterstate(cfg)
    elif agent_type == "ppo_masked":
        train_ppo_masked(cfg)
    else:
        raise ValueError(f"Agent type '{agent_type}' tidak dikenal. Pilih: 'cem', 'td_linear', 'dqn_afterstate', 'ppo_masked'.")
