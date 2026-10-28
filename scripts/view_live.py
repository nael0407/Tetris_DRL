"""Skrip visualisasi langsung (Live View) Tetris RL.

Menjalankan simulasi permainan secara langsung dengan visualisasi grafis (Pygame)
atau terminal teks. Mendukung mode agen acak atau heuristik cerdas sederhana
sehingga pemain dapat melihat animasi pendaratan dan pembersihan baris secara real-time.

Kontrol saat jendela Pygame aktif:
- ESC / Tutup Jendela : Keluar
- SPASI : Jeda / Lanjutkan (Pause / Resume)
- R : Reset episode baru
- PANAH ATAS / BAWAH : Mempercepat / Memperlambat kecepatan animasi (FPS)
"""

import argparse
import sys
import time
import numpy as np
import pygame

from tetris_rl.env import TetrisEnv
from tetris_rl.render_pygame import PygameRenderer
from tetris_rl.render_text import render_board_text


"""Skrip visualisasi langsung (Live View) Tetris RL.

Menjalankan simulasi permainan secara langsung dengan visualisasi grafis (Pygame)
atau terminal teks. Mendukung agen acak, heuristik klasik, CEM, maupun
Deep RL (Deep Q-Network Afterstate / MaskablePPO) sehingga dapat melihat agen
berpikir, menempatkan piece, dan membersihkan baris secara real-time.

Kontrol saat jendela Pygame aktif:
- ESC / Tutup Jendela : Keluar
- SPASI : Jeda / Lanjutkan (Pause / Resume)
- R : Reset episode baru
- PANAH ATAS / BAWAH : Mempercepat / Memperlambat kecepatan animasi (FPS)
"""

import argparse
import os
import sys
import time
from typing import Any, Optional
import numpy as np
import pygame

from tetris_rl.env import TetrisEnv
from tetris_rl.render_pygame import PygameRenderer
from tetris_rl.render_text import render_board_text


def build_live_agent(
    agent_type: str,
    model_path: Optional[str] = None,
    seed: int = 42,
    board_height: int = 20,
    board_width: int = 10,
) -> Any:
    """Membangun instance agen untuk visualisasi live."""
    agent_type = agent_type.lower()
    if agent_type == "random":
        from tetris_rl.agents.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    elif agent_type == "heuristic":
        from tetris_rl.agents.heuristic import HeuristicAgent
        return HeuristicAgent()
    elif agent_type == "cem":
        from tetris_rl.agents.heuristic import HeuristicAgent
        path = model_path or "models/final/cem_features_best.npy"
        weights = np.load(path) if os.path.exists(path) else None
        if weights is not None:
            print(f"[LiveView] Bobot CEM dimuat dari: {path}")
        return HeuristicAgent(weights=weights, name="Trained-CEM")
    elif agent_type in ("dqn", "td_net", "deep_rl"):
        from tetris_rl.agents.td_afterstate import DQNAfterstateAgent
        agent = DQNAfterstateAgent(board_height=board_height, board_width=board_width, representation="raw")
        path = model_path or "models/final/e2_td_net_best.pt"
        if os.path.exists(path):
            agent.load(path)
            agent.epsilon = 0.0
            print(f"[LiveView] Model Deep Q-Network dimuat dari: {path}")
        else:
            print(f"[LiveView] PERINGATAN: Checkpoint '{path}' tidak ditemukan, menggunakan inisialisasi acak.")
        return agent
    elif agent_type in ("ppo", "ppo_masked"):
        from tetris_rl.agents.ppo_masked import MaskedPPOAgent
        path = model_path or "models/final/e3_ppo_masked_best.zip"
        if os.path.exists(path):
            print(f"[LiveView] Model MaskablePPO dimuat dari: {path}")
            return MaskedPPOAgent(model_path=path)
        else:
            raise FileNotFoundError(f"Checkpoint PPO tidak ditemukan: {path}")
    else:
        raise ValueError(f"Agent '{agent_type}' tidak dikenal. Pilih: 'dqn', 'cem', 'heuristic', 'random', 'ppo'.")


def select_agent_action(agent: Any, env: TetrisEnv, obs: np.ndarray, info: dict) -> int:
    """Helper untuk memanggil act dari agen apa pun secara konsisten."""
    from tetris_rl.agents.random_agent import RandomAgent
    from tetris_rl.agents.heuristic import HeuristicAgent
    from tetris_rl.agents.td_afterstate import DQNAfterstateAgent
    from tetris_rl.agents.ppo_masked import MaskedPPOAgent

    if isinstance(agent, DQNAfterstateAgent):
        return agent.act(env=env, obs=obs, info=info, deterministic=True)
    elif isinstance(agent, MaskedPPOAgent):
        return agent.act(env=env, obs=obs, info=info, deterministic=True)
    elif isinstance(agent, HeuristicAgent):
        return agent.act(env=env, obs=obs, info=info)
    elif isinstance(agent, RandomAgent):
        return agent.act(obs=obs, info=info)
    else:
        return agent.act(env=env, obs=obs, info=info)


def run_pygame_live_view(
    agent_type: str = "dqn",
    model_path: Optional[str] = None,
    seed: int = 42,
    fps: int = 15,
    cells_per_frame: float = 1.5,
) -> None:
    """Menjalankan live view interaktif di jendela grafis Pygame."""
    obs_type = "features" if agent_type.lower() in ("heuristic", "cem") else "raw"
    env = TetrisEnv(
        board_height=20,
        board_width=10,
        obs_type=obs_type,
        piece_generator_type="uniform",
        max_steps=10000,
        use_numba=True,
    )

    renderer = PygameRenderer(
        board_height=20,
        board_width=10,
        cell_size=28,
        headless=False,
    )

    agent = build_live_agent(
        agent_type=agent_type,
        model_path=model_path,
        seed=seed,
        board_height=20,
        board_width=10,
    )

    obs, info = env.reset(seed=seed)
    clock = pygame.time.Clock()
    running = True
    paused = False
    current_fps = fps

    print("\n" + "=" * 50)
    print(" TETRIS RL - LIVE VIEW (PYGAME)")
    print("=" * 50)
    print(f" Agen   : {agent_type.upper()}")
    print(f" Seed   : {seed}")
    print(" Kontrol: [SPASI] Pause | [PANAH ATAS/BAWAH] Atur Kecepatan | [ESC] Keluar")
    print("=" * 50 + "\n")

    while running:
        # Polling event GUI
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("[Status] " + ("DIJEDA (PAUSED)" if paused else "DILANJUTKAN"))
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("[Aksi] Reset episode baru!")
                elif event.key == pygame.K_UP:
                    current_fps = min(60, current_fps + 5)
                    print(f"[Kecepatan] Target FPS: {current_fps}")
                elif event.key == pygame.K_DOWN:
                    current_fps = max(2, current_fps - 5)
                    print(f"[Kecepatan] Target FPS: {current_fps}")

        if paused:
            clock.tick(10)
            continue

        # Pilih aksi sesuai agen
        mask = env.action_masks()
        valid_actions = np.nonzero(mask)[0]
        if len(valid_actions) == 0:
            obs, info = env.reset()
            continue

        action = select_agent_action(agent, env, obs, info)

        # Siapkan parameter animasi sebelum step dieksekusi
        board_before = env.board.copy()
        current_p = env.current_piece
        rot = action // env.board_width
        col = action % env.board_width

        # Hitung landing row untuk animasi
        res_preview = env.engine.place_piece(board_before, current_p, rot, col)

        # Mainkan animasi pendaratan piece
        frames = renderer.render_drop_animation(
            board_before=board_before,
            piece_id=current_p,
            rotation=rot,
            col=col,
            landing_row=max(0, res_preview.landing_row),
            board_after=res_preview.board,
            next_piece_id=env.next_piece,
            score=env.score,
            lines_cleared=env.lines_cleared,
            steps=env.steps,
            cells_per_frame=cells_per_frame,
        )

        for frame in frames:
            # Gambarkan frame ke jendela
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            renderer.screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(current_fps * 2)

        # Eksekusi step resmi di environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"[Episode Berakhir] Skor Akhir: {info['score']:,} | Total Baris: {info['lines_cleared']} | Total Langkah: {info['steps']}")
            time.sleep(1.0)
            obs, info = env.reset()

        clock.tick(current_fps)

    renderer.close()
    env.close()
    print("Live View ditutup.")


def run_terminal_live_view(
    agent_type: str = "dqn",
    model_path: Optional[str] = None,
    seed: int = 42,
    max_steps: int = 50,
    delay: float = 0.15,
) -> None:
    """Menjalankan live view di dalam terminal (mode teks)."""
    obs_type = "features" if agent_type.lower() in ("heuristic", "cem") else "raw"
    env = TetrisEnv(board_height=20, board_width=10, obs_type=obs_type, use_numba=True)
    obs, info = env.reset(seed=seed)

    agent = build_live_agent(
        agent_type=agent_type,
        model_path=model_path,
        seed=seed,
        board_height=20,
        board_width=10,
    )

    print("\n" + "=" * 50)
    print(f" TETRIS RL - TERMINAL LIVE VIEW ({agent_type.upper()})")
    print("=" * 50)

    for step in range(max_steps):
        mask = env.action_masks()
        valid_actions = np.nonzero(mask)[0]
        if len(valid_actions) == 0:
            break

        action = select_agent_action(agent, env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        # Bersihkan terminal dan cetak papan terbaru
        sys.stdout.write("\033[2J\033[H")
        txt = render_board_text(
            board=env.board,
            current_piece_id=env.current_piece,
            next_piece_id=env.next_piece,
            score=env.score,
            lines_cleared=env.lines_cleared,
            steps=env.steps,
            use_unicode=True,
        )
        print(txt)
        print(f"\n[Agen: {agent_type.upper()}] Langkah: {step + 1}/{max_steps} | Skor: {info['score']} | Baris Dihapus: {info['lines_cleared']}")

        if terminated or truncated:
            print(f"\nGame Over di langkah {step + 1}!")
            break

        time.sleep(delay)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetris RL Live View")
    parser.add_argument("--mode", type=str, choices=["pygame", "terminal"], default="terminal", help="Mode tampilan (pygame atau terminal)")
    parser.add_argument("--agent", type=str, choices=["dqn", "cem", "heuristic", "random", "ppo"], default="dqn", help="Tipe agen yang dimainkan")
    parser.add_argument("--model", type=str, default=None, help="Path opsional ke model checkpoint (.pt / .npy / .zip)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=50, help="Maksimal langkah pada mode terminal")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS untuk mode pygame")
    parser.add_argument("--delay", type=float, default=0.15, help="Jeda antar langkah (detik) untuk mode terminal")
    args = parser.parse_args()

    if args.mode == "pygame":
        run_pygame_live_view(agent_type=args.agent, model_path=args.model, seed=args.seed, fps=args.fps)
    else:
        run_terminal_live_view(agent_type=args.agent, model_path=args.model, seed=args.seed, max_steps=args.steps, delay=args.delay)

