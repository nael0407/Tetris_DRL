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


def choose_heuristic_action(env: TetrisEnv) -> int:
    """Memilih aksi penempatan terbaik menggunakan heuristik greedy sederhana.

    Memprioritaskan:
    1. Tidak game over.
    2. Menghapus baris sebanyak mungkin.
    3. Posisi pendaratan paling rendah.
    4. Lubang paling sedikit.
    """
    placements = env.valid_placements()
    if not placements:
        return 0

    best_action = placements[0].action
    best_score = -1e9

    for p in placements:
        if p.is_game_over:
            continue
        # Skor penempatan: bonus line clear tinggi, penalti jika ada lubang
        from tetris_rl.features import compute_holes
        holes = compute_holes(p.next_board)
        score = (p.lines_cleared * 1000.0) + (p.landing_row * 10.0) - (holes * 50.0)
        if score > best_score:
            best_score = score
            best_action = p.action

    return best_action


def run_pygame_live_view(
    agent_type: str = "heuristic",
    seed: int = 42,
    fps: int = 15,
    cells_per_frame: float = 1.5,
) -> None:
    """Menjalankan live view interaktif di jendela grafis Pygame."""
    env = TetrisEnv(
        board_height=20,
        board_width=10,
        obs_type="raw",
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

        # Pilih aksi sesuai tipe agen
        mask = env.action_masks()
        valid_actions = np.nonzero(mask)[0]
        if len(valid_actions) == 0:
            obs, info = env.reset()
            continue

        if agent_type == "random":
            action = int(np.random.choice(valid_actions))
        else:
            action = choose_heuristic_action(env)

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
            landing_row=res_preview.landing_row,
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
    agent_type: str = "heuristic",
    seed: int = 42,
    max_steps: int = 50,
    delay: float = 0.15,
) -> None:
    """Menjalankan live view di dalam terminal (mode teks)."""
    env = TetrisEnv(board_height=16, board_width=10, use_numba=True)
    obs, info = env.reset(seed=seed)

    print("\n" + "=" * 50)
    print(" TETRIS RL - TERMINAL LIVE VIEW")
    print("=" * 50)

    for step in range(max_steps):
        mask = env.action_masks()
        valid_actions = np.nonzero(mask)[0]
        if len(valid_actions) == 0:
            break

        if agent_type == "random":
            action = int(np.random.choice(valid_actions))
        else:
            action = choose_heuristic_action(env)

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
        print(f"\nLangkah: {step + 1}/{max_steps} | Skor: {info['score']} | Baris Dihapus: {info['lines_cleared']}")

        if terminated or truncated:
            print("\nGame Over / Selesai!")
            break

        time.sleep(delay)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetris RL Live View")
    parser.add_argument("--mode", type=str, choices=["pygame", "terminal"], default="pygame", help="Mode tampilan (pygame atau terminal)")
    parser.add_argument("--agent", type=str, choices=["heuristic", "random"], default="heuristic", help="Tipe agen yang dimainkan")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS untuk mode pygame")
    args = parser.parse_args()

    if args.mode == "pygame":
        run_pygame_live_view(agent_type=args.agent, seed=args.seed, fps=args.fps)
    else:
        run_terminal_live_view(agent_type=args.agent, seed=args.seed)

