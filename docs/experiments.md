# Katalog Eksperimen Tetris RL (`docs/experiments.md`)

Dokumen ini mencatat seluruh katalog rencana dan realisasi eksperimen sesuai arahan `Agent/DOCUMENTATION_STANDARDS.md`.

---

## Tabel Katalog Eksperimen

| ID | Pertanyaan Riset | Sel Faktorial / Tipe | File Konfigurasi | Perintah Eksekusi | Jumlah Seed | Waktu Jalan | Lokasi Hasil Ringkasan |
|---|---|---|---|---|---|---|---|
| **E1 (Random)** | Seberapa rendah performa baseline batas bawah? | Lower Bound Baseline | `configs/baseline_random.yaml` | `python scripts/evaluate.py --config configs/baseline_random.yaml` | 5 (25 ep) | ~1 detik | `results/summary/baseline_random.json` |
| **E1 (Heuristic)** | Seberapa kuat heuristik berbobot linear klasik (Dellacherie)? | Heuristic Baseline | `configs/baseline_heuristic.yaml` | `python scripts/evaluate.py --config configs/baseline_heuristic.yaml` | 5 (25 ep) | ~6 detik | `results/summary/baseline_heuristic.json` |
| **E1 (CEM)** | Seberapa optimal parameter linear yang dilatih via Cross-Entropy Method? | Baris 1, Kol 1 Faktorial (Gradient-Free x Linear) | `configs/e1_cem_features.yaml` | `python scripts/train.py --config configs/e1_cem_features.yaml` | 5 (25 ep) | ~1.5 menit | `results/summary/e1_cem_features.json` |
| **E2 (TD-Features)** | Apakah gradient-based TD dapat menyamai CEM pada fitur linear? | Baris 2, Kol 1 Faktorial (Gradient-Based x Linear) | `configs/e2_td_features.yaml` | *(Tahap 7-8)* | 5 (25 ep) | *(TBD)* | `results/summary/e2_td_features.json` |
| **E2 (CEM-Net)** | Bagaimana performa neuroevolution CEM pada jaringan papan mentah? | Baris 1, Kol 2 Faktorial (Gradient-Free x Raw Net) | `configs/e2_cem_net.yaml` | *(Tahap 7-8)* | 5 (25 ep) | *(TBD)* | `results/summary/e2_cem_net.json` |
| **E2 (TD-Net / DQN)** | Apakah Deep Q-Network penuh mampu melampaui representasi fitur linear? | Baris 2, Kol 2 Faktorial (Gradient-Based x Raw Net) | `configs/e2_td_net.yaml` | *(Tahap 7-8)* | 5 (25 ep) | *(TBD)* | `results/summary/e2_td_net.json` |
| **E3 (PPO)** | Bagaimana performa Policy Gradient MaskablePPO dibanding sel faktorial? | Outside Factorial (On-Policy Policy Gradient) | `configs/e3_ppo_masked.yaml` | *(Tahap 7-8)* | 5 (25 ep) | *(TBD)* | `results/summary/e3_ppo_masked.json` |
| **E4 (Reward)** | Seberapa penting reward shaping (R0 s.d. R3) terhadap stabilitas belajar? | Variabel Reward | `configs/e4_reward_*.yaml` | *(Tahap 8)* | 5 (25 ep) | *(TBD)* | `results/summary/e4_*.json` |
| **E5 (Afterstate)** | Apakah evaluasi afterstate lebih efisien daripada aksi langsung? | Ruang Transisi | `configs/e5_afterstate.yaml` | *(Tahap 8)* | 5 (25 ep) | *(TBD)* | `results/summary/e5_*.json` |

---

## Log Temuan Eksperimen E1 (Baseline Klasik)

- **Random Agent:** Rata-rata bertahan hidup hanya $20.6$ piece tanpa berhasil membersihkan 1 baris pun (IQM skor 0.0). Membuktikan bahwa game Tetris dengan ruang aksi 40 tidak dapat diselesaikan secara kebetulan.
- **Dellacherie Heuristic (Literature Weights):**
  - Skor IQM: **61,461.5** (95% CI: [39,569.6, 93,086.2]).
  - Baris Terhapus IQM: **558.6** baris per episode.
  - Langkah Bertahan: **1,440.0** piece (Mean: 1,821.5).
  - Rasio Tersensor: **0.0%** (tidak ada yang menyentuh batas 10.000 piece). Ini membuktikan bahwa papan ukuran 10x20 standar bebas dari *ceiling effect* untuk baseline heuristik ini.
- **Cross-Entropy Method (CEM) on Features:**
  - Total Piece Interaksi Pelatihan: **659,779 piece**.
  - Bobot Optimal Ditemukan: $[-0.87491, 0.44129, -0.06376, -0.18903]$.
  - Skor Evaluasi IQM: **55,169.2** (95% CI: [34,792.3, 89,749.2]).
  - Baris Terhapus IQM: **498.6** baris per episode.
  - Hasil menunjukkan bahwa CEM yang dioptimasi secara mandiri dari nol berhasil mencapai performa yang setara/mendekati bobot literatur Dellacherie dalam 660k piece interaksi.

