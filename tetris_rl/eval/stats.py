"""Modul statistik evaluasi standar riset RL (IQM, Bootstrap CI, Survival Curve).

Sesuai Dokumen Proyek Bagian 3.5, 5.3, dan 7:
- Interquartile Mean (IQM) dan 95% Bootstrap Confidence Intervals (Agarwal et al., 2021).
- Estimasi kurva survival Kaplan-Meier untuk pelaporan data tersensor (batas 10.000 piece).
- Proporsi run yang mencapai batas episode (censored run rate).
"""

from typing import Dict, List, NamedTuple, Tuple, Union
import numpy as np
import scipy.stats as stats
import rliable.metrics as rmetrics


class EvalStats(NamedTuple):
    """Ringkasan statistik metrik evaluasi."""
    mean: float
    std: float
    median: float
    iqm: float
    ci_low: float
    ci_high: float
    censored_ratio: float


def compute_iqm(data: Union[List[float], np.ndarray]) -> float:
    """Menghitung Interquartile Mean (rata-rata 50% data di antara kuartil 25% dan 75%).

    Metrik ini lebih tangguh (robust) terhadap outlier ekstrem dibanding mean aritmetika.
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) < 4:
        return float(np.mean(arr))
    # Menggunakan scipy trim_mean dengan proporsi 0.25 di kedua sisi (total 50% terpotong)
    return float(stats.trim_mean(arr, proportiontocut=0.25))


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Menghitung interval kepercayaan (confidence interval) bootstrap untuk IQM.

    Args:
        data: Array sampel metrik.
        n_boot: Jumlah iterasi bootstrap (default 2000).
        ci: Tingkat kepercayaan (default 0.95 untuk CI 95%).
        seed: Seed bilangan acak untuk reproduktibilitas.

    Returns:
        Tuple[float, float]: (batas_bawah, batas_atas).
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) < 3:
        m = float(np.mean(arr))
        return m, m

    rng = np.random.default_rng(seed)
    boot_iqms = np.empty(n_boot, dtype=np.float64)

    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_iqms[i] = compute_iqm(sample)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_iqms, alpha * 100))
    high = float(np.percentile(boot_iqms, (1.0 - alpha) * 100))
    return low, high


def compute_censored_ratio(
    steps_list: Union[List[int], np.ndarray],
    limit: int = 10000,
) -> float:
    """Menghitung proporsi run yang mencapai batas maksimal (tersensor)."""
    arr = np.asarray(steps_list)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr >= limit))


def compute_kaplan_meier_survival(
    durations: Union[List[int], np.ndarray],
    censored_limit: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Menghitung kurva survival empiris Kaplan-Meier untuk data tersensor.

    Args:
        durations: Durasi langkah bertahan (steps survived) untuk tiap episode.
        censored_limit: Batas episode pemotongan.

    Returns:
        timeline: Titik-titik waktu langkah unik.
        survival_prob: Probabilitas bertahan hidup P(T >= t).
    """
    arr = np.asarray(durations, dtype=np.int64)
    if len(arr) == 0:
        return np.array([0]), np.array([1.0])

    # Status: 1 jika mati (game over sebelum batas), 0 jika tersensor (mencapai batas)
    events = (arr < censored_limit).astype(np.int32)

    # Urutkan berdasarkan waktu
    order = np.argsort(arr)
    sorted_durations = arr[order]
    sorted_events = events[order]

    unique_times, counts = np.unique(sorted_durations, return_counts=True)
    n_at_risk = len(arr)
    survival = 1.0

    timeline = [0]
    probabilities = [1.0]

    for t in unique_times:
        mask = (sorted_durations == t)
        d_i = np.sum(sorted_events[mask])  # Jumlah kematian pada waktu t
        n_i = n_at_risk                    # Jumlah yang berisiko sebelum waktu t

        if n_i > 0 and d_i > 0:
            survival *= (1.0 - d_i / n_i)

        timeline.append(int(t))
        probabilities.append(float(survival))

        n_at_risk -= len(mask)

    return np.array(timeline), np.array(probabilities)


def summarize_metric(
    data: Union[List[float], np.ndarray],
    censored_limit: int = 10000,
) -> EvalStats:
    """Menghitung seluruh ringkasan statistik metrik secara terintegrasi."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return EvalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    iqm_val = compute_iqm(arr)
    ci_l, ci_h = compute_bootstrap_ci(arr)
    c_ratio = compute_censored_ratio(arr, limit=censored_limit)

    return EvalStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        median=float(np.median(arr)),
        iqm=iqm_val,
        ci_low=ci_l,
        ci_high=ci_h,
        censored_ratio=c_ratio,
    )

