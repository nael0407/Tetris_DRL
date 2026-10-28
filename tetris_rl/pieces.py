"""Definisi 7 tetromino dan rotasi uniknya untuk Tetris RL.

Modul ini mendefinisikan representasi matematis dari ke-7 tetromino standar (I, O, T, S, Z, J, L).
Sesuai Dokumen Proyek (Bagian 3.1 & 4.2), hanya rotasi unik yang dimodelkan:
- O: 1 rotasi unik (simetri simetris)
- I, S, Z: 2 rotasi unik (simetri 180 derajat)
- T, J, L: 4 rotasi unik (tanpa simetri rotasi)

Total konfigurasi rotasi valid adalah 19.
Setiap konfigurasi memiliki bounding box rapat (tight bounding box) dan tepat 4 sel aktif.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Konstanta global ruang aksi & board default
BOARD_WIDTH_DEFAULT: int = 10
BOARD_HEIGHT_DEFAULT: int = 20
MAX_ROTATIONS: int = 4
NUM_PIECES: int = 7
ACTION_SPACE_SIZE: int = MAX_ROTATIONS * BOARD_WIDTH_DEFAULT  # 40 aksi diskrit (0-39)

# Mapping nama dan ID tetromino
PIECE_NAMES: Tuple[str, ...] = ("I", "O", "T", "S", "Z", "J", "L")
PIECE_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(PIECE_NAMES)}
PIECE_ID_TO_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(PIECE_NAMES)}

# Jumlah rotasi unik per tetromino sesuai Dokumen Proyek Bagian 3.1
PIECE_ROTATIONS_COUNT: Tuple[int, ...] = (2, 1, 4, 2, 2, 4, 4)


@dataclass(frozen=True)
class PieceRotation:
    """Representasi satu orientasi rotasi dari tetromino.

    Attributes:
        rotation_idx: Indeks rotasi (0 sampai num_rotations - 1).
        shape: Matriks biner 2D uint8 rapat (h, w) di mana 1 adalah blok terisi.
        coords: Koordinat sel terisi (4, 2) bertipe int32 sebagai offset (row, col)
                relatif terhadap pojok kiri atas bounding box [0, 0].
        height: Tinggi bounding box piece dalam blok.
        width: Lebar bounding box piece dalam blok.
    """
    rotation_idx: int
    shape: np.ndarray
    coords: np.ndarray
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.shape.ndim != 2:
            raise ValueError(f"Shape piece harus berdimensi 2, didapat {self.shape.ndim}")
        if np.sum(self.shape) != 4:
            raise ValueError(f"Tetromino harus memiliki tepat 4 sel aktif, didapat {np.sum(self.shape)}")
        if self.coords.shape != (4, 2):
            raise ValueError(f"Coords piece harus berbentuk (4, 2), didapat {self.coords.shape}")


@dataclass(frozen=True)
class Piece:
    """Definisi lengkap satu jenis tetromino beserta seluruh rotasi uniknya."""
    id: int
    name: str
    num_rotations: int
    rotations: Tuple[PieceRotation, ...]

    def __post_init__(self) -> None:
        if len(self.rotations) != self.num_rotations:
            raise ValueError(
                f"Jumlah rotasi terdaftar ({len(self.rotations)}) tidak sama dengan "
                f"num_rotations ({self.num_rotations}) untuk piece {self.name}"
            )


def _build_rotation(rotation_idx: int, grid_list: List[List[int]]) -> PieceRotation:
    """Helper internal untuk membuat objek PieceRotation dari daftar 2D int."""
    shape_arr = np.array(grid_list, dtype=np.uint8)
    coords_list: List[Tuple[int, int]] = []
    for r in range(shape_arr.shape[0]):
        for c in range(shape_arr.shape[1]):
            if shape_arr[r, c] == 1:
                coords_list.append((r, c))
    coords_arr = np.array(coords_list, dtype=np.int32)
    return PieceRotation(
        rotation_idx=rotation_idx,
        shape=shape_arr,
        coords=coords_arr,
        height=int(shape_arr.shape[0]),
        width=int(shape_arr.shape[1]),
    )


# Definisi bentuk matriks biner untuk tiap tetromino
# Catatan: Tiap bentuk dipangkas rapat (tight bounding box) tanpa baris/kolom kosong ekstra.
_RAW_SHAPES: Dict[str, List[List[List[int]]]] = {
    # I: 2 rotasi unik (horizontal & vertical)
    "I": [
        [[1, 1, 1, 1]],                    # rot 0: horizontal (1x4)
        [[1], [1], [1], [1]],              # rot 1: vertikal (4x1)
    ],
    # O: 1 rotasi unik (simetris)
    "O": [
        [[1, 1],
         [1, 1]],                          # rot 0: kotak (2x2)
    ],
    # T: 4 rotasi unik (arah bawah, kiri, atas, kanan)
    "T": [
        [[1, 1, 1],
         [0, 1, 0]],                       # rot 0: menunjuk ke bawah (2x3)
        [[0, 1],
         [1, 1],
         [0, 1]],                          # rot 1: menunjuk ke kiri (3x2)
        [[0, 1, 0],
         [1, 1, 1]],                       # rot 2: menunjuk ke atas (2x3)
        [[1, 0],
         [1, 1],
         [1, 0]],                          # rot 3: menunjuk ke kanan (3x2)
    ],
    # S: 2 rotasi unik (horizontal & vertical)
    "S": [
        [[0, 1, 1],
         [1, 1, 0]],                       # rot 0: horizontal (2x3)
        [[1, 0],
         [1, 1],
         [0, 1]],                          # rot 1: vertikal (3x2)
    ],
    # Z: 2 rotasi unik (horizontal & vertical)
    "Z": [
        [[1, 1, 0],
         [0, 1, 1]],                       # rot 0: horizontal (2x3)
        [[0, 1],
         [1, 1],
         [1, 0]],                          # rot 1: vertikal (3x2)
    ],
    # J: 4 rotasi unik
    "J": [
        [[1, 0, 0],
         [1, 1, 1]],                       # rot 0: hook kiri-atas (2x3)
        [[1, 1],
         [1, 0],
         [1, 0]],                          # rot 1: hook kanan-atas (3x2)
        [[1, 1, 1],
         [0, 0, 1]],                       # rot 2: hook kanan-bawah (2x3)
        [[0, 1],
         [0, 1],
         [1, 1]],                          # rot 3: hook kiri-bawah (3x2)
    ],
    # L: 4 rotasi unik
    "L": [
        [[0, 0, 1],
         [1, 1, 1]],                       # rot 0: hook kanan-atas (2x3)
        [[1, 0],
         [1, 0],
         [1, 1]],                          # rot 1: hook kanan-bawah (3x2)
        [[1, 1, 1],
         [1, 0, 0]],                       # rot 2: hook kiri-bawah (2x3)
        [[1, 1],
         [0, 1],
         [0, 1]],                          # rot 3: hook kiri-atas (3x2)
    ],
}


def _construct_all_pieces() -> Tuple[Piece, ...]:
    pieces: List[Piece] = []
    for name in PIECE_NAMES:
        p_id = PIECE_NAME_TO_ID[name]
        raw_rots = _RAW_SHAPES[name]
        num_rots = len(raw_rots)
        rots: List[PieceRotation] = [
            _build_rotation(r_idx, grid)
            for r_idx, grid in enumerate(raw_rots)
        ]
        pieces.append(
            Piece(
                id=p_id,
                name=name,
                num_rotations=num_rots,
                rotations=tuple(rots),
            )
        )
    return tuple(pieces)


# Objek pieces terkonstruksi
PIECES: Tuple[Piece, ...] = _construct_all_pieces()

# Struktur lookup cepat berukuran statis untuk optimasi Numba & NumPy di tahap selanjutnya
# COORDS_TABLE: (7, 4, 4, 2) int32 -> [piece_id, rot_idx, block_idx, (r, c)]
# Nilai -1 digunakan sebagai sentinel untuk rotasi di luar num_rotations
COORDS_TABLE: np.ndarray = np.full((NUM_PIECES, MAX_ROTATIONS, 4, 2), -1, dtype=np.int32)
# DIMS_TABLE: (7, 4, 2) int32 -> [piece_id, rot_idx, (height, width)]
DIMS_TABLE: np.ndarray = np.zeros((NUM_PIECES, MAX_ROTATIONS, 2), dtype=np.int32)
# NUM_ROTATIONS_TABLE: (7,) int32
NUM_ROTATIONS_TABLE: np.ndarray = np.array(PIECE_ROTATIONS_COUNT, dtype=np.int32)

for p in PIECES:
    for rot in p.rotations:
        COORDS_TABLE[p.id, rot.rotation_idx] = rot.coords
        DIMS_TABLE[p.id, rot.rotation_idx, 0] = rot.height
        DIMS_TABLE[p.id, rot.rotation_idx, 1] = rot.width


def get_piece(identifier: Union[int, str]) -> Piece:
    """Mengambil objek Piece berdasarkan ID integer (0-6) atau nama string ('I', 'O', dst)."""
    if isinstance(identifier, str):
        if identifier not in PIECE_NAME_TO_ID:
            raise KeyError(f"Nama piece tidak dikenal: {identifier}. Pilihan valid: {PIECE_NAMES}")
        identifier = PIECE_NAME_TO_ID[identifier]
    if not (0 <= identifier < NUM_PIECES):
        raise IndexError(f"ID piece {identifier} di luar jangkauan [0, {NUM_PIECES - 1}]")
    return PIECES[identifier]


def get_piece_rotation(piece_id: int, rotation_idx: int) -> PieceRotation:
    """Mengambil objek PieceRotation spesifik berdasarkan piece_id dan indeks rotasi."""
    p = get_piece(piece_id)
    if not (0 <= rotation_idx < p.num_rotations):
        raise IndexError(
            f"Rotasi {rotation_idx} tidak valid untuk piece {p.name}. "
            f"Piece ini hanya memiliki {p.num_rotations} rotasi unik (indeks 0..{p.num_rotations - 1})."
        )
    return p.rotations[rotation_idx]



# --- Generator Urutan Piece (Randomizer) ---


class PieceGenerator(ABC):
    """Kelas abstrak dasar untuk pembangkit urutan tetromino."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Mereset kondisi acak generator dengan seed tertentu."""
        pass

    @abstractmethod
    def next_piece(self) -> int:
        """Menghasilkan ID piece berikutnya (0 sampai 6)."""
        pass


class UniformPieceGenerator(PieceGenerator):
    """Pengacak piece seragam independen (default penelitian Tetris RL).

    Setiap langkah membangkitkan ID piece (0..6) secara acak seragam.
    Semua keacakan dikontrol oleh instance np.random.Generator untuk determinisme total.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def next_piece(self) -> int:
        return int(self.rng.integers(0, NUM_PIECES))


class SevenBagPieceGenerator(PieceGenerator):
    """Pengacak 7-bag standar Tetris Guideline.

    Setiap kantong (bag) berisi permutasi acak dari ke-7 tetromino unik (I, O, T, S, Z, J, L).
    Piece baru diambil dari kantong hingga habis, lalu kantong baru dikocok.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.current_bag: List[int] = []
        self._refill()

    def _refill(self) -> None:
        perm = self.rng.permutation(NUM_PIECES).tolist()
        self.current_bag = [int(x) for x in perm]

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_bag.clear()
        self._refill()

    def next_piece(self) -> int:
        if not self.current_bag:
            self._refill()
        return self.current_bag.pop(0)


def create_piece_generator(
    generator_type: str = "uniform",
    seed: Optional[int] = None,
) -> PieceGenerator:
    """Factory function untuk membuat generator piece.

    Args:
        generator_type: Jenis pengacak ('uniform' untuk acak seragam, '7-bag' untuk 7-bag).
        seed: Seed bilangan acak untuk reproduktibilitas.

    Returns:
        Instance PieceGenerator.
    """
    gen_type = generator_type.lower()
    if gen_type in ("uniform", "random"):
        return UniformPieceGenerator(seed=seed)
    elif gen_type in ("7-bag", "7bag", "seven_bag", "sevenbag"):
        return SevenBagPieceGenerator(seed=seed)
    else:
        raise ValueError(
            f"Tipe generator piece '{generator_type}' tidak dikenal. Pilihan valid: 'uniform', '7-bag'."
        )
