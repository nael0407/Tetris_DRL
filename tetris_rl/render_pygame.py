"""Tampilan grafis Pygame untuk Tetris RL dengan animasi pendaratan piece.

Modul ini mengimplementasikan visualizer grafis sesuai prinsip Dokumen Proyek Bagian 9:
- Terpisah penuh dari logika permainan (tidak pernah menggambar di dalam loop pelatihan).
- Mendukung mode headless (dummy video driver) untuk perekaman rekaman GIF/MP4 tanpa monitor.
- Menganimasikan piece yang meluncur jatuh (hard drop) ke posisi pendaratan secara halus
  sebagai efek visual murni, sementara logika permainan tetap satu langkah per piece.
- Menampilkan panel informasi: Skor, Baris Dihapus, Langkah, Piece Aktif, dan Preview Next Piece.
"""

import os
from typing import List, Optional, Tuple, Union
import numpy as np
import pygame
import imageio

from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    PIECE_ID_TO_NAME,
    PIECE_NAMES,
    PieceRotation,
    get_piece,
    get_piece_rotation,
)

# Palet warna modern Tetris (RGB)
COLORS = {
    "bg": (18, 18, 24),
    "board_bg": (28, 28, 38),
    "grid_line": (42, 42, 58),
    "border": (75, 75, 105),
    "panel_bg": (24, 24, 34),
    "text_primary": (240, 240, 245),
    "text_secondary": (160, 160, 185),
    "text_accent": (0, 220, 220),
    "ghost_alpha": 70,
    "flash": (255, 255, 255),
}

# Warna spesifik per jenis tetromino
PIECE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 220, 220),    # I: Cyan
    (240, 215, 0),    # O: Yellow
    (170, 50, 230),   # T: Purple
    (50, 205, 50),    # S: Green
    (235, 50, 50),    # Z: Red
    (40, 100, 235),   # J: Blue
    (245, 140, 20),   # L: Orange
)


class PygameRenderer:
    """Renderer visualisasi grafis Tetris dengan dukungan mode headless dan animasi pendaratan."""

    def __init__(
        self,
        board_height: int = BOARD_HEIGHT_DEFAULT,
        board_width: int = BOARD_WIDTH_DEFAULT,
        cell_size: int = 30,
        margin: int = 20,
        side_panel_width: int = 190,
        headless: bool = False,
    ) -> None:
        """Inisialisasi renderer Pygame.

        Args:
            board_height: Tinggi papan dalam blok (default 20).
            board_width: Lebar papan dalam blok (default 10).
            cell_size: Ukuran satu blok persegi dalam piksel (default 30).
            margin: Margin tepi jendela dalam piksel.
            side_panel_width: Lebar panel statistik di samping kanan papan.
            headless: Jika True, gunakan driver dummy video tanpa membuat jendela fisik OS.
        """
        self.board_height = board_height
        self.board_width = board_width
        self.cell_size = cell_size
        self.margin = margin
        self.side_panel_width = side_panel_width
        self.headless = headless

        # Jika headless atau tidak ada DISPLAY terdeteksi di Linux, gunakan dummy driver
        if self.headless or "DISPLAY" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # Dimensi permukaan
        self.board_pixel_w = self.board_width * self.cell_size
        self.board_pixel_h = self.board_height * self.cell_size

        self.screen_width = self.margin + self.board_pixel_w + self.margin + self.side_panel_width + self.margin
        self.screen_height = self.margin + self.board_pixel_h + self.margin

        if self.headless:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Tetris RL - Visualizer")

        # Font
        self.font_title = pygame.font.SysFont("dejavusans", 18, bold=True)
        self.font_label = pygame.font.SysFont("dejavusans", 12)
        self.font_val = pygame.font.SysFont("dejavusans", 15, bold=True)

    def _draw_block(
        self,
        surface: pygame.Surface,
        pixel_x: int,
        pixel_y: int,
        color: Tuple[int, int, int],
        alpha: int = 255,
        is_ghost: bool = False,
    ) -> None:
        """Helper untuk menggambar satu sel blok dengan efek bevel / rounded corner."""
        s = self.cell_size
        if is_ghost:
            ghost_surf = pygame.Surface((s, s), pygame.SRCALPHA)
            ghost_color = (*color, alpha)
            pygame.draw.rect(ghost_surf, ghost_color, (1, 1, s - 2, s - 2), width=2, border_radius=3)
            surface.blit(ghost_surf, (pixel_x, pixel_y))
            return

        rect = pygame.Rect(pixel_x, pixel_y, s, s)
        # Warna blok utama
        pygame.draw.rect(surface, color, (pixel_x + 1, pixel_y + 1, s - 2, s - 2), border_radius=3)

        # Highlight kilau atas-kiri untuk efek 3D arcade
        highlight_color = tuple(min(255, c + 50) for c in color)
        pygame.draw.line(surface, highlight_color, (pixel_x + 2, pixel_y + 2), (pixel_x + s - 3, pixel_y + 2), 2)
        pygame.draw.line(surface, highlight_color, (pixel_x + 2, pixel_y + 2), (pixel_x + 2, pixel_y + s - 3), 2)

        # Shadow bawah-kanan
        shadow_color = tuple(max(0, c - 60) for c in color)
        pygame.draw.line(surface, shadow_color, (pixel_x + 2, pixel_y + s - 2), (pixel_x + s - 2, pixel_y + s - 2), 2)
        pygame.draw.line(surface, shadow_color, (pixel_x + s - 2, pixel_y + 2), (pixel_x + s - 2, pixel_y + s - 2), 2)

    def _draw_mini_piece(
        self,
        surface: pygame.Surface,
        piece_id: int,
        center_x: int,
        center_y: int,
        mini_size: int = 18,
    ) -> None:
        """Menggambar miniatur tetromino untuk preview."""
        p_rot = get_piece_rotation(piece_id, 0)
        p_w = p_rot.width * mini_size
        p_h = p_rot.height * mini_size
        start_x = center_x - p_w // 2
        start_y = center_y - p_h // 2
        color = PIECE_COLORS[piece_id]

        for dr, dc in p_rot.coords:
            bx = start_x + dc * mini_size
            by = start_y + dr * mini_size
            pygame.draw.rect(surface, color, (bx + 1, by + 1, mini_size - 2, mini_size - 2), border_radius=2)

    def render_frame(
        self,
        board: np.ndarray,
        active_piece_id: Optional[int] = None,
        active_rotation: int = 0,
        active_row: Optional[float] = None,
        active_col: Optional[int] = None,
        ghost_landing_row: Optional[int] = None,
        next_piece_id: Optional[int] = None,
        score: int = 0,
        lines_cleared: int = 0,
        steps: int = 0,
        flash_rows: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Merender satu frame tampilan lengkap dan mengembalikan array RGB (H, W, 3)."""
        self.screen.fill(COLORS["bg"])

        # 1. Gambar Area Papan (Board Surface)
        b_x = self.margin
        b_y = self.margin
        b_w = self.board_pixel_w
        b_h = self.board_pixel_h

        # Latar belakang papan & border
        pygame.draw.rect(self.screen, COLORS["board_bg"], (b_x, b_y, b_w, b_h))
        pygame.draw.rect(self.screen, COLORS["border"], (b_x - 2, b_y - 2, b_w + 4, b_h + 4), width=2, border_radius=4)

        # Garis grid halus
        for r in range(1, self.board_height):
            y = b_y + r * self.cell_size
            pygame.draw.line(self.screen, COLORS["grid_line"], (b_x, y), (b_x + b_w, y), 1)
        for c in range(1, self.board_width):
            x = b_x + c * self.cell_size
            pygame.draw.line(self.screen, COLORS["grid_line"], (x, b_y), (x, b_y + b_h), 1)

        # 2. Gambar Blok yang Sudah Ada di Papan
        for r in range(self.board_height):
            # Efek flash jika baris sedang terhapus
            if flash_rows and r in flash_rows:
                pygame.draw.rect(
                    self.screen,
                    COLORS["flash"],
                    (b_x, b_y + r * self.cell_size, b_w, self.cell_size),
                )
                continue

            for c in range(self.board_width):
                val = board[r, c]
                if val != 0:
                    px = b_x + c * self.cell_size
                    py = b_y + r * self.cell_size
                    # Default cyan jika biner 1, atau sesuai nilai ID jika bernilai 1..7
                    c_idx = (int(val) - 1) % len(PIECE_COLORS) if val > 1 else 0
                    self._draw_block(self.screen, px, py, PIECE_COLORS[c_idx])

        # 3. Gambar Ghost Piece (Bayangan Pendaratan)
        if (
            active_piece_id is not None
            and active_col is not None
            and ghost_landing_row is not None
            and ghost_landing_row >= 0
        ):
            p_rot = get_piece_rotation(active_piece_id, active_rotation)
            g_color = PIECE_COLORS[active_piece_id]
            for dr, dc in p_rot.coords:
                gr = ghost_landing_row + dr
                gc = active_col + dc
                if 0 <= gr < self.board_height and 0 <= gc < self.board_width:
                    gx = b_x + gc * self.cell_size
                    gy = b_y + gr * self.cell_size
                    self._draw_block(self.screen, gx, gy, g_color, is_ghost=True)

        # 4. Gambar Active Piece (Sedang Jatuh / Melayang)
        if (
            active_piece_id is not None
            and active_col is not None
            and active_row is not None
        ):
            p_rot = get_piece_rotation(active_piece_id, active_rotation)
            p_color = PIECE_COLORS[active_piece_id]
            for dr, dc in p_rot.coords:
                exact_r = active_row + dr
                exact_c = active_col + dc
                if 0 <= exact_c < self.board_width:
                    px = b_x + exact_c * self.cell_size
                    py = int(b_y + exact_r * self.cell_size)
                    if py >= b_y - self.cell_size and py < b_y + b_h:
                        self._draw_block(self.screen, px, py, p_color)

        # 5. Gambar Side Panel (Statistik & Preview)
        p_x = b_x + b_w + self.margin
        p_y = self.margin
        p_w = self.side_panel_width
        p_h = self.board_pixel_h

        pygame.draw.rect(self.screen, COLORS["panel_bg"], (p_x, p_y, p_w, p_h), border_radius=6)
        pygame.draw.rect(self.screen, COLORS["border"], (p_x, p_y, p_w, p_h), width=1, border_radius=6)

        # Judul
        title_surf = self.font_title.render("TETRIS RL", True, COLORS["text_accent"])
        self.screen.blit(title_surf, (p_x + 15, p_y + 15))

        # Helper kartu statistik
        def draw_stat(lbl: str, val_str: str, top_y: int) -> None:
            lbl_surf = self.font_label.render(lbl, True, COLORS["text_secondary"])
            val_surf = self.font_val.render(val_str, True, COLORS["text_primary"])
            self.screen.blit(lbl_surf, (p_x + 15, top_y))
            self.screen.blit(val_surf, (p_x + 15, top_y + 16))

        draw_stat("SCORE", f"{score:,}", p_y + 55)
        draw_stat("LINES", str(lines_cleared), p_y + 105)
        draw_stat("STEPS", str(steps), p_y + 155)

        # Box Next Piece Preview
        next_box_y = p_y + 215
        pygame.draw.rect(self.screen, COLORS["board_bg"], (p_x + 15, next_box_y, p_w - 30, 80), border_radius=4)
        next_lbl = self.font_label.render("NEXT PIECE", True, COLORS["text_secondary"])
        self.screen.blit(next_lbl, (p_x + 20, next_box_y + 8))

        if next_piece_id is not None:
            self._draw_mini_piece(
                self.screen,
                next_piece_id,
                center_x=p_x + p_w // 2,
                center_y=next_box_y + 48,
            )

        # Box Active Piece Preview
        act_box_y = next_box_y + 95
        pygame.draw.rect(self.screen, COLORS["board_bg"], (p_x + 15, act_box_y, p_w - 30, 80), border_radius=4)
        act_lbl = self.font_label.render("CURRENT PIECE", True, COLORS["text_secondary"])
        self.screen.blit(act_lbl, (p_x + 20, act_box_y + 8))

        if active_piece_id is not None:
            self._draw_mini_piece(
                self.screen,
                active_piece_id,
                center_x=p_x + p_w // 2,
                center_y=act_box_y + 48,
            )

        # Render ke array numpy RGB (H, W, 3)
        rgb_array = pygame.surfarray.array3d(self.screen)
        # pygame.surfarray mengembalikan (W, H, 3) sehingga perlu transpose ke (H, W, 3)
        return np.transpose(rgb_array, (1, 0, 2))

    def render_drop_animation(
        self,
        board_before: np.ndarray,
        piece_id: int,
        rotation: int,
        col: int,
        landing_row: int,
        board_after: np.ndarray,
        next_piece_id: Optional[int] = None,
        score: int = 0,
        lines_cleared: int = 0,
        steps: int = 0,
        cells_per_frame: float = 2.0,
    ) -> List[np.ndarray]:
        """Menganimasikan piece meluncur jatuh ke posisi penempatan akhir dan line-clear flash.

        Sesuai Dokumen Proyek Bagian 9: Animasi ini murni representasi visual untuk perekaman
        evolusi belajar agen dan demo, tanpa mengubah logika permainan 1 langkah per piece.

        Returns:
            List[np.ndarray]: Daftar frame gambar RGB (H, W, 3).
        """
        frames: List[np.ndarray] = []
        p_rot = get_piece_rotation(piece_id, rotation)
        spawn_row = 0.0

        # 1. Frame animasi meluncur jatuh dari atas ke landing_row
        current_r = spawn_row
        target_r = float(max(0, landing_row))

        while current_r < target_r:
            frame = self.render_frame(
                board=board_before,
                active_piece_id=piece_id,
                active_rotation=rotation,
                active_row=current_r,
                active_col=col,
                ghost_landing_row=landing_row,
                next_piece_id=next_piece_id,
                score=score,
                lines_cleared=lines_cleared,
                steps=steps,
            )
            frames.append(frame)
            current_r += cells_per_frame

        # Frame pendaratan tepat di posisi akhir
        frame_landed = self.render_frame(
            board=board_before,
            active_piece_id=piece_id,
            active_rotation=rotation,
            active_row=target_r,
            active_col=col,
            ghost_landing_row=landing_row,
            next_piece_id=next_piece_id,
            score=score,
            lines_cleared=lines_cleared,
            steps=steps,
        )
        frames.append(frame_landed)

        # 2. Cek apakah ada baris yang terhapus
        diff = board_after.astype(int) - board_before.astype(int)
        # Baris terhapus adalah baris yang pada board_after kosong sedangkan sebelumnya penuh
        # atau kita tampilkan frame hasil board_after
        frame_after = self.render_frame(
            board=board_after,
            next_piece_id=next_piece_id,
            score=score,
            lines_cleared=lines_cleared,
            steps=steps,
        )
        frames.append(frame_after)

        return frames

    def save_gif(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 20,
    ) -> None:
        """Menyimpan urutan frame RGB menjadi file animasi GIF menggunakan imageio."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Hitung duration per frame dalam milidetik
        duration_ms = int(1000 / fps)
        imageio.mimsave(output_path, frames, duration=duration_ms, loop=0)

    def close(self) -> None:
        """Menutup instance pygame dengan bersih."""
        pygame.quit()

