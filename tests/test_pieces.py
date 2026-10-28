"""Unit tests untuk modul tetris_rl.pieces.

Memverifikasi kepatuhan terhadap spesifikasi Tetris RL:
- Tepat 7 jenis tetromino standar (I, O, T, S, Z, J, L).
- Jumlah rotasi unik sesuai: O: 1, I/S/Z: 2, T/J/L: 4.
- Tepat 4 sel aktif per konfigurasi rotasi.
- Koordinat konsisten dengan matriks biner dan dimensi bounding box.
- Tidak ada rotasi duplikat pada piece yang sama.
- Validasi struktur tabel Numba/NumPy (COORDS_TABLE, DIMS_TABLE).
"""

import numpy as np
import pytest

from tetris_rl.pieces import (
    BOARD_HEIGHT_DEFAULT,
    BOARD_WIDTH_DEFAULT,
    ACTION_SPACE_SIZE,
    COORDS_TABLE,
    DIMS_TABLE,
    NUM_PIECES,
    NUM_ROTATIONS_TABLE,
    PIECE_NAMES,
    PIECE_NAME_TO_ID,
    PIECE_ROTATIONS_COUNT,
    PIECES,
    Piece,
    PieceRotation,
    get_piece,
    get_piece_rotation,
)


def test_piece_count_and_names() -> None:
    """Memastikan ada tepat 7 piece dengan nama standar."""
    assert len(PIECES) == 7
    assert NUM_PIECES == 7
    expected_names = ("I", "O", "T", "S", "Z", "J", "L")
    assert PIECE_NAMES == expected_names
    assert len(PIECE_NAME_TO_ID) == 7
    for idx, p in enumerate(PIECES):
        assert p.id == idx
        assert p.name == expected_names[idx]


def test_rotation_counts() -> None:
    """Memastikan jumlah rotasi unik per piece tepat sesuai Dokumen Proyek Bagian 3.1."""
    expected_rotations = {
        "O": 1,
        "I": 2,
        "S": 2,
        "Z": 2,
        "T": 4,
        "J": 4,
        "L": 4,
    }
    assert PIECE_ROTATIONS_COUNT == (2, 1, 4, 2, 2, 4, 4)
    for p in PIECES:
        assert p.num_rotations == expected_rotations[p.name]
        assert len(p.rotations) == expected_rotations[p.name]
        assert NUM_ROTATIONS_TABLE[p.id] == expected_rotations[p.name]


def test_each_rotation_has_exactly_four_cells() -> None:
    """Memastikan setiap rotasi memiliki tepat 4 blok terisi pada matriks dan koordinat."""
    total_rotations = 0
    for p in PIECES:
        for rot in p.rotations:
            total_rotations += 1
            # Cek pada matriks shape
            assert rot.shape.dtype == np.uint8
            assert np.sum(rot.shape) == 4, f"{p.name} rotasi {rot.rotation_idx} tidak punya 4 sel"
            # Cek pada array koordinat
            assert rot.coords.shape == (4, 2)
            assert rot.coords.dtype == np.int32
            # Koordinat harus unik (tidak ada sel bertumpuk)
            unique_coords = {tuple(c) for c in rot.coords}
            assert len(unique_coords) == 4

    # Total rotasi unik di seluruh 7 piece: 1 + 2 + 2 + 2 + 4 + 4 + 4 = 19
    assert total_rotations == 19


def test_bounding_box_and_coordinate_consistency() -> None:
    """Memastikan dimensi bounding box rapat dan koordinat cocok dengan nilai 1 pada matriks."""
    for p in PIECES:
        for rot in p.rotations:
            h, w = rot.shape.shape
            assert rot.height == h
            assert rot.width == w

            # Pastikan tidak ada baris atau kolom kosong di tepi (tight bounding box)
            assert np.any(rot.shape[0, :] == 1), f"Baris teratas {p.name} rot {rot.rotation_idx} kosong"
            assert np.any(rot.shape[h - 1, :] == 1), f"Baris terbawah {p.name} rot {rot.rotation_idx} kosong"
            assert np.any(rot.shape[:, 0] == 1), f"Kolom terkiri {p.name} rot {rot.rotation_idx} kosong"
            assert np.any(rot.shape[:, w - 1] == 1), f"Kolom terkanan {p.name} rot {rot.rotation_idx} kosong"

            # Cocokkan koordinat dengan matriks biner
            reconstructed_shape = np.zeros((h, w), dtype=np.uint8)
            for r, c in rot.coords:
                assert 0 <= r < h
                assert 0 <= c < w
                reconstructed_shape[r, c] = 1
            np.testing.assert_array_equal(rot.shape, reconstructed_shape)


def test_no_duplicate_rotations_within_piece() -> None:
    """Memastikan semua orientasi rotasi dalam satu piece benar-benar unik secara geometri."""
    for p in PIECES:
        seen_shapes = []
        for rot in p.rotations:
            for prev_shape in seen_shapes:
                if rot.shape.shape == prev_shape.shape:
                    assert not np.array_equal(rot.shape, prev_shape), (
                        f"Piece {p.name} memiliki rotasi duplikat pada indeks {rot.rotation_idx}"
                    )
            seen_shapes.append(rot.shape)


def test_get_piece_and_rotation_accessors() -> None:
    """Memastikan fungsi pembantu get_piece dan get_piece_rotation bekerja dengan benar."""
    # Akses dengan nama dan id
    assert get_piece("I").name == "I"
    assert get_piece(0).name == "I"
    assert get_piece("O").id == 1
    assert get_piece("L").id == 6

    # Error handling id / nama salah
    with pytest.raises(KeyError):
        get_piece("INVALID")
    with pytest.raises(IndexError):
        get_piece(-1)
    with pytest.raises(IndexError):
        get_piece(7)

    # Akses rotasi
    rot0 = get_piece_rotation(piece_id=0, rotation_idx=0)
    assert rot0.rotation_idx == 0
    assert rot0.width == 4
    assert rot0.height == 1

    rot1 = get_piece_rotation(piece_id=0, rotation_idx=1)
    assert rot1.rotation_idx == 1
    assert rot1.width == 1
    assert rot1.height == 4

    # Rotasi di luar batas unik harus menghasilkan IndexError
    with pytest.raises(IndexError):
        get_piece_rotation(piece_id=0, rotation_idx=2)  # 'I' hanya punya rotasi 0 dan 1
    with pytest.raises(IndexError):
        get_piece_rotation(piece_id=1, rotation_idx=1)  # 'O' hanya punya rotasi 0


def test_static_tables_consistency() -> None:
    """Memastikan COORDS_TABLE dan DIMS_TABLE konsisten dengan objek PIECES."""
    assert COORDS_TABLE.shape == (7, 4, 4, 2)
    assert DIMS_TABLE.shape == (7, 4, 2)

    for p in PIECES:
        for rot_idx in range(4):
            if rot_idx < p.num_rotations:
                expected_coords = p.rotations[rot_idx].coords
                expected_dims = (p.rotations[rot_idx].height, p.rotations[rot_idx].width)
                np.testing.assert_array_equal(COORDS_TABLE[p.id, rot_idx], expected_coords)
                np.testing.assert_array_equal(DIMS_TABLE[p.id, rot_idx], expected_dims)
            else:
                # Rotasi yang tidak valid harus berisi sentinel -1 (coords) dan 0 (dims)
                assert np.all(COORDS_TABLE[p.id, rot_idx] == -1)
                assert np.all(DIMS_TABLE[p.id, rot_idx] == 0)


def test_constants() -> None:
    """Memastikan konstanta dasar board dan ruang aksi sesuai Dokumen Proyek."""
    assert BOARD_WIDTH_DEFAULT == 10
    assert BOARD_HEIGHT_DEFAULT == 20
    assert ACTION_SPACE_SIZE == 40

