"""Unit tests untuk modul statistik evaluasi (tetris_rl.eval.stats).

Memverifikasi:
- Ketepatan perhitungan Interquartile Mean (IQM) terhadap data acak dan outlier ekstrem.
- Interval kepercayaan bootstrap (95% CI) mencakup nilai IQM.
- Proporsi data tersensor (censored proportion).
- Monotonisitas non-increasing pada kurva survival Kaplan-Meier.
"""

import numpy as np
import pytest

from tetris_rl.eval.stats import (
    compute_bootstrap_ci,
    compute_censored_ratio,
    compute_iqm,
    compute_kaplan_meier_survival,
    summarize_metric,
)


def test_compute_iqm() -> None:
    """Menguji ketahanan IQM terhadap outlier ekstrem."""
    # Data bersih: [10, 20, 30, 40, 50, 60, 70, 80] -> kuartil tengah adalah 30, 40, 50, 60 -> rata-rata 45
    data_clean = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    iqm_clean = compute_iqm(data_clean)
    assert 40.0 <= iqm_clean <= 50.0

    # Tambahkan outlier sangat besar (misal 1.000.000)
    data_outlier = data_clean + [1_000_000.0, 2_000_000.0]
    mean_val = np.mean(data_outlier)
    iqm_val = compute_iqm(data_outlier)

    # Mean biasa melonjak drastis, tapi IQM tetap stabil
    assert mean_val > 100_000.0
    assert iqm_val < 100.0


def test_bootstrap_ci() -> None:
    """Memastikan batas bootstrap CI valid dan mengapit nilai IQM."""
    rng = np.random.default_rng(seed=123)
    data = rng.normal(loc=100.0, scale=10.0, size=50)

    iqm_val = compute_iqm(data)
    ci_low, ci_high = compute_bootstrap_ci(data, n_boot=1000, ci=0.95, seed=42)

    assert ci_low <= iqm_val <= ci_high
    assert 85.0 < ci_low < 100.0
    assert 100.0 < ci_high < 115.0


def test_censored_ratio() -> None:
    """Menguji proporsi episode yang mencapai batas maksimum 10.000 langkah."""
    steps = [500, 10000, 2300, 10000, 4500]  # 2 dari 5 tersensor = 40%
    ratio = compute_censored_ratio(steps, limit=10000)
    assert ratio == pytest.approx(0.4)


def test_kaplan_meier_survival() -> None:
    """Memverifikasi bahwa kurva survival Kaplan-Meier bernilai [0, 1] dan monoton turun."""
    durations = [150, 400, 1200, 10000, 10000]
    timeline, survival_prob = compute_kaplan_meier_survival(durations, censored_limit=10000)

    assert timeline[0] == 0
    assert survival_prob[0] == 1.0
    # Probabilitas harus non-increasing
    for i in range(len(survival_prob) - 1):
        assert survival_prob[i] >= survival_prob[i + 1]
    assert np.all(survival_prob >= 0.0)
    assert np.all(survival_prob <= 1.0)


def test_summarize_metric() -> None:
    """Menguji struktur kembalian EvalStats."""
    data = [100, 200, 300, 400, 500, 10000]
    stats_res = summarize_metric(data, censored_limit=10000)
    assert stats_res.mean > 0
    assert stats_res.iqm > 0
    assert stats_res.ci_low <= stats_res.iqm <= stats_res.ci_high
    assert stats_res.censored_ratio == pytest.approx(1.0 / 6.0)

