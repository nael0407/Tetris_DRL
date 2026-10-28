"""Skrip perekaman visualisasi gameplay agen ke file animasi GIF (record.py).

Merekam performa agen (misal agen acak untuk video baseline, heuristik,
atau CEM hasil pelatihan) ke dalam file animasi GIF dengan visualisasi
animasi pendaratan piece yang mulus.

Penggunaan:
    python scripts/record.py
    python scripts/record.py --agent random --output docs/assets/demo_random.gif
    python scripts/record.py --agent heuristic --output docs/assets/demo_heuristic.gif
    python scripts/record.py --agent cem --weights models/final/cem_features_best.npy --output docs/assets/demo_cem.gif
"""

import argparse
import os
from typing import Optional, Union
import numpy as np

from tetris_rl.agents.heuristic import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.render_pygame import PygameRenderer


def record_agent_gameplay(
    agent_type: str = "random",
    weights_path: Optional[str] = None,
    output_path: str = "docs/assets/demo.gif",
    max_steps: int = 40,
    seed: int = 42,
    fps: int = 15,
    board_height: int = 20,
    board_width: int = 10,
) -> None:
    """Merekam gameplay agen dan menyimpannya ke file animasi GIF.

    Args:
        agent_type: Tipe agen ('random', 'heuristic', 'cem').
        weights_path: Path ke file bobot (.npy) jika menggunakan CEM.
        output_path: Path tujuan penyimpanan file GIF.
        max_steps: Jumlah langkah interaksi maksimum yang direkam.
        seed: Random seed untuk determinisme.
        fps: Frame per second untuk file animasi GIF.
        board_height: Tinggi papan Tetris (default: 20).
        board_width: Lebar papan Tetris (default: 10).
    """
    env = TetrisEnv(
        board_height=board_height,
        board_width=board_width,
        obs_type="raw" if agent_type == "random" else "features",
        use_numba=True,
    )

    renderer = PygameRenderer(
        board_height=board_height,
        board_width=board_width,
        cell_size=24,
        headless=True,
    )

    # Inisialisasi agen sesuai tipe
    agent_type_lower = agent_type.lower()
    agent: Union[RandomAgent, HeuristicAgent]
    if agent_type_lower == "random":
        agent = RandomAgent(seed=seed)
    elif agent_type_lower in ("heuristic", "cem"):
        weights = None
        # Auto-discovery bobot CEM default jika tidak diberikan secara eksplisit
        if weights_path is None and agent_type_lower == "cem":
            default_cem_path = "models/final/cem_features_best.npy"
            if os.path.exists(default_cem_path):
                weights_path = default_cem_path

        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"File bobot tidak ditemukan: {weights_path}")
            weights = np.load(weights_path)
            print(f"[Record] Memuat bobot dari: {weights_path}")
        else:
            if agent_type_lower == "cem":
                print("[Record] PERINGATAN: Bobot CEM tidak ditentukan. Menggunakan bobot heuristik standar.")

        agent = HeuristicAgent(weights=weights)
    else:
        raise ValueError(f"Agent '{agent_type}' tidak dikenal. Pilih: 'random', 'heuristic', 'cem'.")

    obs, info = env.reset(seed=seed)
    all_frames = []

    print(f"[Record] Merekam gameplay {agent_type_lower} (maks {max_steps} langkah, papan {board_width}x{board_height})...")

    for step in range(max_steps):
        board_before = env.board.copy()
        current_p = env.current_piece

        if isinstance(agent, HeuristicAgent):
            action = agent.act(env=env, obs=obs, info=info)
        else:
            action = agent.act(obs=obs, info=info)

        rot = action // env.board_width
        col = action % env.board_width

        res_preview = env.engine.place_piece(board_before, current_p, rot, col)
        effective_landing_row = max(0, res_preview.landing_row)

        frames = renderer.render_drop_animation(
            board_before=board_before,
            piece_id=current_p,
            rotation=rot,
            col=col,
            landing_row=effective_landing_row,
            board_after=res_preview.board,
            next_piece_id=env.next_piece,
            score=env.score,
            lines_cleared=env.lines_cleared,
            steps=env.steps,
            cells_per_frame=2.0,
        )
        all_frames.extend(frames)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"[Record] Episode berakhir di langkah {step + 1}!")
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    renderer.save_gif(all_frames, output_path, fps=fps)
    renderer.close()
    env.close()

    file_size = os.path.getsize(output_path)
    print(f"[Record] Berhasil menyimpan rekaman ke {output_path} ({len(all_frames)} frames, {file_size:,} bytes).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rekam Gameplay Agen Tetris RL ke Animasi GIF")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["random", "heuristic", "cem"],
        default="random",
        help="Tipe agen yang direkam (random, heuristic, atau cem)",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path ke file bobot .npy jika ada")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/assets/demo.gif",
        help="Path file keluaran GIF (default: docs/assets/demo.gif)",
    )
    parser.add_argument("--steps", type=int, default=30, help="Jumlah langkah maksimum yang direkam")
    parser.add_argument("--seed", type=int, default=42, help="Seed evaluasi")
    parser.add_argument("--fps", type=int, default=15, help="Frame per second GIF")
    parser.add_argument("--height", type=int, default=20, help="Tinggi papan Tetris (default: 20)")
    parser.add_argument("--width", type=int, default=10, help="Lebar papan Tetris (default: 10)")
    args = parser.parse_args()

    record_agent_gameplay(
        agent_type=args.agent,
        weights_path=args.weights,
        output_path=args.output,
        max_steps=args.steps,
        seed=args.seed,
        fps=args.fps,
        board_height=args.height,
        board_width=args.width,
    )


