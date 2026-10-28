"""Unit tests untuk modul visualisasi Tetris RL (render_text & render_pygame).

Memverifikasi kriteria Tahap 4 Dokumen Proyek:
- Tampilan teks di terminal (ASCII & Unicode) merender papan dan statistik dengan benar.
- Renderer Pygame (mode headless) menghasilkan frame RGB berdimensi valid.
- Animasi pendaratan piece (drop animation) menghasilkan urutan frame mulus.
- Ekspor rekaman GIF menggunakan imageio berhasil dan dapat dibaca ulang.
"""

import os
import tempfile
import numpy as np
import pytest
import imageio

import tetris_rl.board as board_module
from tetris_rl.pieces import PIECE_NAME_TO_ID, get_piece
from tetris_rl.render_pygame import PygameRenderer
from tetris_rl.render_text import print_board, render_board_text


def test_render_board_text_unicode_and_ascii() -> None:
    """Menguji modul render_text dalam mode Unicode dan ASCII."""
    board = board_module.create_board(height=10, width=6)
    board[9, 1:5] = 1  # Baris bawah berisi blok

    # 1. Mode Unicode
    txt_unicode = render_board_text(
        board,
        current_piece_id=0,
        next_piece_id=1,
        score=1500,
        lines_cleared=4,
        steps=12,
        use_unicode=True,
    )
    assert "TETRIS RL" in txt_unicode
    assert "Score : 1,500" in txt_unicode
    assert "Lines : 4" in txt_unicode
    assert "Steps : 12" in txt_unicode
    assert "Piece : I" in txt_unicode
    assert "Next  : O" in txt_unicode
    assert "┌" in txt_unicode and "┘" in txt_unicode
    assert "■" in txt_unicode

    # 2. Mode ASCII murni
    txt_ascii = render_board_text(
        board,
        current_piece_id=2,
        next_piece_id=3,
        use_unicode=False,
    )
    assert "+" in txt_ascii
    assert "#" in txt_ascii
    assert "Piece : T" in txt_ascii


def test_render_board_text_overlay() -> None:
    """Menguji penempatan piece overlay pada teks papan."""
    board = board_module.create_board(height=10, width=6)
    # Overlay piece I horizontal di row 2 col 1
    txt = render_board_text(
        board,
        overlay_piece_id=0,
        overlay_rotation=0,
        overlay_row=2,
        overlay_col=1,
        use_unicode=True,
    )
    # Harus ada karakter overlay '▣'
    assert "▣" in txt


def test_pygame_renderer_frame_shape_and_headless() -> None:
    """Memastikan PygameRenderer dapat berjalan di mode headless dan menghasilkan frame valid."""
    renderer = PygameRenderer(
        board_height=20,
        board_width=10,
        cell_size=20,
        headless=True,
    )

    board = board_module.create_board(height=20, width=10)
    board[19, :] = 1

    frame = renderer.render_frame(
        board=board,
        active_piece_id=0,
        active_rotation=0,
        active_row=5.0,
        active_col=3,
        ghost_landing_row=18,
        next_piece_id=1,
        score=800,
        lines_cleared=1,
        steps=5,
    )

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    # Dimensi harus (H, W, 3)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.shape[0] == renderer.screen_height
    assert frame.shape[1] == renderer.screen_width

    renderer.close()


def test_pygame_drop_animation_and_gif_export() -> None:
    """Menguji pembuatan animasi jatuh dan ekspor rekaman GIF menggunakan imageio."""
    renderer = PygameRenderer(
        board_height=12,
        board_width=8,
        cell_size=20,
        headless=True,
    )

    board_before = board_module.create_board(height=12, width=8)
    # Drop piece O di kolom 2
    id_o = PIECE_NAME_TO_ID["O"]
    res = board_module.place_piece(board_before, piece_id=id_o, rotation=0, col=2)

    # Bangkitkan animasi frame jatuh
    frames = renderer.render_drop_animation(
        board_before=board_before,
        piece_id=id_o,
        rotation=0,
        col=2,
        landing_row=res.landing_row,
        board_after=res.board,
        next_piece_id=PIECE_NAME_TO_ID["I"],
        score=100,
        lines_cleared=0,
        steps=1,
        cells_per_frame=2.0,
    )

    assert len(frames) >= 4, f"Animasi harus menghasilkan beberapa frame, didapat {len(frames)}"

    # Simpan ke file GIF sementara dan verifikasi
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = os.path.join(tmpdir, "test_drop.gif")
        renderer.save_gif(frames, gif_path, fps=15)

        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 1000  # File GIF berisi data visual

        # Baca kembali menggunakan imageio untuk memvalidasi integritas file GIF
        loaded_reader = imageio.get_reader(gif_path)
        loaded_frames = [f for f in loaded_reader]
        assert len(loaded_frames) == len(frames)
        assert loaded_frames[0].shape == frames[0].shape

    renderer.close()

