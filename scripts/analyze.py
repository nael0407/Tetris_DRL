"""Skrip Analisis Interpretabilitas (analyze.py).

Mengeksekusi eksperimen interpretabilitas pada model DQN terlatih (Tahap 9):
1. Linear Probing: Mengukur seberapa baik representasi laten DQN mengodekan
   konsep lubang, ketinggian, bumpiness, dan sumur (R^2 score).
2. Intervensi Papan: Mengukur perubahan fungsi nilai Delta V saat disuntikkan
   lubang buatan atau dinaikkan ketinggian kolom.
3. Menghasilkan grafik visual dan menyimpan ringkasan hasil ke results/analysis/.

Penggunaan:
    python scripts/analyze.py --model models/final/e2_td_net_best.pt
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from tetris_rl.agents.td_afterstate import DQNAfterstateAgent
from tetris_rl.analysis.intervention import run_intervention_analysis
from tetris_rl.analysis.probing import collect_probing_dataset, run_linear_probing
from tetris_rl.env import TetrisEnv


def run_interpretability_pipeline(
    model_path: str = "models/final/e2_td_net_best.pt",
    output_dir: str = "results/analysis",
    assets_dir: str = "docs/assets",
    num_samples: int = 1200,
    seed: int = 42,
) -> None:
    """Menjalankan pipeline analisis interpretabilitas penuh."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(" MEMULAI ANALISIS INTERPRETABILITAS TETRIS RL (TAHAP 9)")
    print("=" * 65)

    env = TetrisEnv(board_height=20, board_width=10, use_numba=True)
    agent = DQNAfterstateAgent(
        board_height=20,
        board_width=10,
        representation="raw",
        hidden_dims=(128, 64),
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint tidak ditemukan: {model_path}")

    print(f"[Model] Memuat model DQN dari: {model_path}")
    agent.load(model_path)

    # 1. Linear Probing
    print(f"\n[1/3] Menjalankan Linear Probing pada {num_samples} sampel representasi...")
    probing_results = run_linear_probing(
        agent=agent,
        env=env,
        num_samples=num_samples,
        layer_idx=0,
        seed=seed,
    )

    probing_json = {
        feat: {
            "r2_score": float(res.r2_score),
            "pearson_r": float(res.pearson_r),
            "mse": float(res.mse),
        }
        for feat, res in probing_results.items()
    }
    probing_out_path = os.path.join(output_dir, "probing_results.json")
    with open(probing_out_path, "w", encoding="utf-8") as f:
        json.dump(probing_json, f, indent=2)

    print("[Linear Probing] Hasil R^2 per fitur laten:")
    for feat, res in probing_results.items():
        print(f"  - {feat:<18}: R^2 = {res.r2_score:6.3f} | Pearson r = {res.pearson_r:6.3f}")
    print(f"  -> Disimpan di: {probing_out_path}")

    # 2. Intervensi Papan Terkontrol
    print(f"\n[2/3] Menjalankan Intervensi Papan Terkontrol (Hole & Height Injection)...")
    boards, _ = collect_probing_dataset(env, num_samples=300, seed=seed)
    interv_results = run_intervention_analysis(agent, [b for b in boards], rng=np.random.default_rng(seed))

    interv_json = {
        k: {
            "mean_delta_value": float(v.mean_delta_value),
            "std_delta_value": float(v.std_delta_value),
            "fraction_negative_impact": float(v.fraction_negative_impact),
            "num_samples": int(v.num_samples),
        }
        for k, v in interv_results.items()
    }
    interv_out_path = os.path.join(output_dir, "intervention_results.json")
    with open(interv_out_path, "w", encoding="utf-8") as f:
        json.dump(interv_json, f, indent=2)

    print("[Intervensi Kausal] Efek Delta V:")
    for k, v in interv_results.items():
        print(f"  - {k:<18}: Mean Delta V = {v.mean_delta_value:7.3f} | Proporsi Negatif: {v.fraction_negative_impact*100:5.1f}%")
    print(f"  -> Disimpan di: {interv_out_path}")

    # 3. Pembuatan Grafik Visualisasi
    print(f"\n[3/3] Menghasilkan grafik visualisasi untuk laporan dan dokumentasi...")
    # Grafik 1: Probing R^2 Bar Chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    features = list(probing_results.keys())
    r2_vals = [probing_results[f].r2_score for f in features]
    colors = ["#2b5c8f", "#3c7bb6", "#569bd8", "#7fb8e8", "#a6d2f7"]

    bars = ax.bar(features, r2_vals, color=colors, edgecolor="#1c3b5e", width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_title("Linear Probing R^2 Scores on DQN Latent Representations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Linear Probe Test R^2 Score", fontsize=10)
    ax.set_xlabel("Ground-Truth Geometric Features", fontsize=10)

    for bar, val in zip(bars, r2_vals):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart1_path = os.path.join(assets_dir, "probing_r2_scores.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"  -> Grafik Probing R^2 disimpan di: {chart1_path}")

    # Grafik 2: Intervention Delta V Bar Chart
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    int_names = ["Hole Injection", "Height Increase (+2)"]
    int_means = [interv_results["hole_injection"].mean_delta_value, interv_results["height_increase"].mean_delta_value]
    int_stds = [interv_results["hole_injection"].std_delta_value, interv_results["height_increase"].std_delta_value]

    bar_colors = ["#d9534f" if m < 0 else "#5cb85c" for m in int_means]
    bars2 = ax.bar(int_names, int_means, yerr=int_stds, capsize=5, color=bar_colors, edgecolor="#333333", width=0.45)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Causal Board Interventions: Mean Delta V(s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Delta V(s) = V(Intervened) - V(Original)", fontsize=10)

    for bar, val in zip(bars2, int_means):
        offset = 0.05 if val >= 0 else -0.15
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + offset, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart2_path = os.path.join(assets_dir, "intervention_value_delta.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"  -> Grafik Intervensi Nilai disimpan di: {chart2_path}")

    print("\n" + "=" * 65)
    print(" ANALISIS INTERPRETABILITAS SELESAI DENGAN SUKSES!")
    print("=" * 65 + "\n")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis Interpretabilitas Tetris RL")
    parser.add_argument("--model", type=str, default="models/final/e2_td_net_best.pt", help="Path ke model DQN terlatih")
    parser.add_argument("--samples", type=int, default=1200, help="Jumlah sampel papan untuk probing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_interpretability_pipeline(
        model_path=args.model,
        num_samples=args.samples,
        seed=args.seed,
    )

