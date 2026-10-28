"""Modul Linear Probing untuk Interpretabilitas Representasi Laten (probing.py).

Sesuai Dokumen Proyek Bagian 8:
- Menjawab pertanyaan: "Konsep apa yang sebenarnya dikodekan oleh hidden layer jaringan saraf tiruan?"
- Melatih regresi linear (linear probe) dari aktivasi hidden layer jaringan DQN
  ke fitur-fitur fisik papan terstruktur (holes, max_height, aggregate_height, bumpiness).
- Jika probe linear mencapai R^2 yang tinggi, ini membuktikan bahwa jaringan secara mandiri
  menemukan dan merepresentasikan konsep struktural tersebut dalam representasi latennya.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from tetris_rl.agents.td_afterstate import DQNAfterstateAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.features import (
    compute_aggregate_height,
    compute_bumpiness,
    compute_holes,
    compute_max_height,
)


class ProbingResult(NamedTuple):
    """Hasil evaluasi satu fitur pada linear probing."""
    feature_name: str
    r2_score: float
    pearson_r: float
    mse: float


def compute_wells(board: np.ndarray) -> int:
    """Menghitung kedalaman sumur (well) pada papan Tetris.

    Sumur adalah kolom vertikal kosong yang diapit oleh blok di kiri dan kanan (atau dinding).
    """
    H, W = board.shape
    total_wells = 0

    for c in range(W):
        for r in range(H):
            if board[r, c] == 0:
                # Cek apakah kiri tertutup (atau dinding kiri)
                left_closed = (c == 0) or (board[r, c - 1] != 0)
                # Cek apakah kanan tertutup (atau dinding kanan)
                right_closed = (c == W - 1) or (board[r, c + 1] != 0)
                if left_closed and right_closed:
                    total_wells += 1
            else:
                break
    return total_wells


def extract_ground_truth_features(board: np.ndarray) -> Dict[str, float]:
    """Mengekstrak fitur-fitur dasar terstruktur dari papan biner."""
    return {
        "holes": float(compute_holes(board)),
        "max_height": float(compute_max_height(board)),
        "aggregate_height": float(compute_aggregate_height(board)),
        "bumpiness": float(compute_bumpiness(board)),
        "wells": float(compute_wells(board)),
    }


def extract_hidden_activations(
    agent: DQNAfterstateAgent,
    boards: np.ndarray,
    layer_idx: int = 0,
) -> np.ndarray:
    """Mengekstrak vektor aktivasi dari hidden layer tertentu pada model DQN.

    Args:
        agent: DQNAfterstateAgent terlatih.
        boards: Array papan berdimensi (N, H, W) atau (N, 200).
        layer_idx: Indeks hidden layer (0 = hidden layer pertama setelah ReLU).

    Returns:
        np.ndarray: Matriks aktivasi berdimensi (N, hidden_dim).
    """
    agent.policy_net.eval()
    if boards.ndim == 3:
        N = boards.shape[0]
        flat_boards = boards.reshape(N, -1).astype(np.float32)
    else:
        flat_boards = boards.astype(np.float32)

    tensor_in = torch.from_numpy(flat_boards).to(agent.device)

    # Identifikasi layer linear di Sequential: [Linear, ReLU, Linear, ReLU, Linear]
    # Linear 0 -> ReLU 1 -> Linear 2 -> ReLU 3 -> Linear 4
    target_module_idx = 1 if layer_idx == 0 else 3
    activations: List[torch.Tensor] = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    seq = agent.policy_net.network
    target_module = seq[target_module_idx] if target_module_idx < len(seq) else seq[1]
    handle = target_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = agent.policy_net(tensor_in)

    handle.remove()
    return activations[0].numpy()


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    l2_reg: float = 1.0,
) -> ProbingResult:
    """Melatih least-squares / ridge regression probe dan menguji R^2 pada data test."""
    N_train, D = X_train.shape
    # Tambahkan bias (intercept)
    X_train_bias = np.hstack([X_train, np.ones((N_train, 1), dtype=np.float32)])
    X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)])

    # Solusi Ridge: w = (X^T X + lambda * I)^(-1) X^T y
    reg_matrix = l2_reg * np.eye(D + 1, dtype=np.float32)
    reg_matrix[-1, -1] = 0.0  # Jangan meregularisasi bias

    A = X_train_bias.T @ X_train_bias + reg_matrix
    b = X_train_bias.T @ y_train
    weights = np.linalg.solve(A, b)

    # Prediksi pada test set
    y_pred = X_test_bias @ weights
    mse = float(np.mean((y_test - y_pred) ** 2))

    # Hitung R^2
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    # Pearson correlation r
    if np.std(y_pred) > 1e-8 and np.std(y_test) > 1e-8:
        pearson_r = float(np.corrcoef(y_test, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    return ProbingResult(
        feature_name="",
        r2_score=max(0.0, float(r2)),
        pearson_r=float(pearson_r),
        mse=mse,
    )


def collect_probing_dataset(
    env: TetrisEnv,
    num_samples: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Mengumpulkan dataset variasi papan dan label ground-truth dari permainan simulasi."""
    rng = np.random.default_rng(seed)
    boards: List[np.ndarray] = []
    features_dict: Dict[str, List[float]] = {
        "holes": [],
        "max_height": [],
        "aggregate_height": [],
        "bumpiness": [],
        "wells": [],
    }

    obs, info = env.reset(seed=seed)

    while len(boards) < num_samples:
        placements = env.valid_placements()
        if not placements:
            obs, info = env.reset()
            continue

        # Simpan state papan saat ini
        b = env.board.copy()
        gt = extract_ground_truth_features(b)
        boards.append(b)
        for k, v in gt.items():
            features_dict[k].append(v)

        # Pilih aksi acak legal untuk memperkaya variasi tumpukan
        act = rng.choice([p.action for p in placements])
        obs, reward, term, trunc, info = env.step(act)
        if term or trunc:
            obs, info = env.reset()

    boards_arr = np.array(boards[:num_samples], dtype=np.uint8)
    labels = {k: np.array(v[:num_samples], dtype=np.float32) for k, v in features_dict.items()}
    return boards_arr, labels


def run_linear_probing(
    agent: DQNAfterstateAgent,
    env: TetrisEnv,
    num_samples: int = 1500,
    test_ratio: float = 0.25,
    layer_idx: int = 0,
    seed: int = 42,
) -> Dict[str, ProbingResult]:
    """Menjalankan analisis linear probing penuh terhadap model DQN terlatih."""
    boards, labels = collect_probing_dataset(env, num_samples=num_samples, seed=seed)
    activations = extract_hidden_activations(agent, boards, layer_idx=layer_idx)

    # Pembagian train/test split
    num_test = int(num_samples * test_ratio)
    indices = np.arange(num_samples)
    np.random.default_rng(seed).shuffle(indices)

    test_idx = indices[:num_test]
    train_idx = indices[num_test:]

    X_train = activations[train_idx]
    X_test = activations[test_idx]

    results: Dict[str, ProbingResult] = {}
    for feat_name, y in labels.items():
        y_train = y[train_idx]
        y_test = y[test_idx]
        res = fit_linear_probe(X_train, y_train, X_test, y_test)
        results[feat_name] = ProbingResult(
            feature_name=feat_name,
            r2_score=res.r2_score,
            pearson_r=res.pearson_r,
            mse=res.mse,
        )

    return results

