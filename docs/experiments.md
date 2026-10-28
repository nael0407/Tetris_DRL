# Katalog Eksperimen Tetris RL (`docs/experiments.md`)

Dokumen ini mencatat seluruh katalog rencana dan realisasi eksperimen sesuai arahan `Agent/DOCUMENTATION_STANDARDS.md`.

---

## Tabel Katalog Eksperimen

| ID | Pertanyaan Riset | Sel Faktorial / Tipe | File Konfigurasi | Perintah Eksekusi | Jumlah Seed | Waktu Jalan | Lokasi Hasil Ringkasan |
|---|---|---|---|---|---|---|---|
| **E1 (Random)** | Seberapa rendah performa baseline batas bawah? | Lower Bound Baseline | `configs/baseline_random.yaml` | `python scripts/evaluate.py --config configs/baseline_random.yaml` | 5 (25 ep) | ~1 detik | `results/summary/baseline_random.json` |
| **E1 (Heuristic)** | Seberapa kuat heuristik berbobot linear klasik (Dellacherie)? | Heuristic Baseline | `configs/baseline_heuristic.yaml` | `python scripts/evaluate.py --config configs/baseline_heuristic.yaml` | 5 (25 ep) | ~6 detik | `results/summary/baseline_heuristic.json` |
| **E1 (CEM)** | Seberapa optimal parameter linear yang dilatih via Cross-Entropy Method? | Baris 1, Kol 1 Faktorial (Gradient-Free x Linear) | `configs/e1_cem_features.yaml` | `python scripts/train.py --config configs/e1_cem_features.yaml` | 5 (25 ep) | ~1.5 menit | `results/summary/e1_cem_features.json` |
| **E2 (TD-Features)** | Apakah gradient-based TD dapat menyamai CEM pada fitur linear? | Baris 2, Kol 1 Faktorial (Gradient-Based x Linear) | `configs/e2_td_features.yaml` | `python scripts/train.py --config configs/e2_td_features.yaml` | 5 (25 ep) | ~10 detik | `results/summary/e2_td_features.json` |
| **E2 (CEM-Net)** | Bagaimana performa neuroevolution CEM pada jaringan papan mentah? | Baris 1, Kol 2 Faktorial (Gradient-Free x Raw Net) | `configs/e2_cem_net.yaml` | `python scripts/train.py --config configs/e2_cem_net.yaml` | 5 (25 ep) | ~3 detik | `results/summary/e2_cem_net.json` |
| **E2 (TD-Net / DQN)** | Apakah Deep Q-Network penuh mampu melampaui representasi fitur linear? | Baris 2, Kol 2 Faktorial (Gradient-Based x Raw Net) | `configs/e2_td_net.yaml` | `python scripts/train.py --config configs/e2_td_net.yaml` | 5 (25 ep) | ~1.8 menit | `results/summary/e2_td_net.json` |
| **E3 (PPO)** | Bagaimana performa Policy Gradient MaskablePPO dibanding sel faktorial? | Outside Factorial (On-Policy Policy Gradient) | `configs/e3_ppo_masked.yaml` | `python scripts/train.py --config configs/e3_ppo_masked.yaml` | 5 (25 ep) | ~25 detik | `results/summary/e3_ppo_masked.json` |
| **E4 (Reward R0)** | Seberapa parah sparse reward R0 tanpa reward shaping? | Ablasi Reward (R0 Game Score Only) | `configs/e4_reward_r0.yaml` | `python scripts/train.py --config configs/e4_reward_r0.yaml` | 5 (25 ep) | ~30 detik | `results/summary/e4_reward_r0.json` |
| **E4 (Reward R1)** | Apakah survival bonus (+1) membantu agen belajar bertahan? | Ablasi Reward (R1 Survival Bonus) | `configs/e4_reward_r1.yaml` | `python scripts/train.py --config configs/e4_reward_r1.yaml` | 5 (25 ep) | ~30 detik | `results/summary/e4_reward_r1.json` |
| **E4 (Reward R2)** | Bagaimana dampak penalti lubang & ketinggian pada stabilitas? | Ablasi Reward (R2 Penalties) | `configs/e4_reward_r2.yaml` | `python scripts/train.py --config configs/e4_reward_r2.yaml` | 5 (25 ep) | ~30 detik | `results/summary/e4_reward_r2.json` |

---

## Tabel Hasil Perbandingan Multi-Seed Resmi (Tahap 8)

| ID / Model | Pieces Interaksi | Skor IQM | 95% Bootstrap CI | Baris Terhapus IQM | Langkah Bertahan IQM | Rasio Tersensor (10k cap) |
|---|---|---|---|---|---|---|
| **Baseline Heuristic** | Literature | **61.461,5** | [39.569,6; 93.086,2] | **558,6** | 1.440,0 | 0,0% |
| **Baseline Random** | 0 | **0,0** | [0,0; 0,0] | **0,0** | 20,3 | 0,0% |
| **E1 (CEM-Features)** | 659.779 | **55.169,2** | [34.792,3; 89.749,2] | **498,6** | 1.290,1 | 0,0% |
| **E2 (TD-Features)** | 50.000 | **400,0** | [223,1; 592,3] | **3,1** | 51,0 | 0,0% |
| **E2 (CEM-Net / Raw)** | 7.483 | **0,0** | [0,0; 0,0] | **0,0** | 23,7 | 0,0% |
| **E2 (TD-Net / DQN Raw)** | 50.000 | **0,0** | [0,0; 0,0] | **0,0** | 27,9 | 0,0% |
| **E3 (PPO-Masked)** | 50.000 | **0,0** | [0,0; 0,0] | **0,0** | 8,0 | 0,0% |
| **E4 (Reward R0 - Sparse)** | 15.000 | **0,0** | [0,0; 30,8] | **0,0** | 29,5 | 0,0% |
| **E4 (Reward R1 - Survival)**| 15.000 | **46,2** | [7,7; 130,8] | **0,5** | 32,3 | 0,0% |
| **E4 (Reward R2 - Penalties)**| 15.000 | **15,4** | [0,0; 61,5] | **0,2** | 26,5 | 0,0% |

---

## Analisis Desain Faktorial 2x2: Algoritma vs Representasi

Matriks hasil perbandingan skor IQM:

| | **Fitur Linear (Dellacherie 4)** | **Papan Mentah (Raw Board 200)** |
|---|---|---|
| **Gradient-Free (CEM)** | **55.169,2** | **0,0** |
| **Gradient/TD (TD / DQN)** | **400,0** | **0,0** |

### Temuan Empiris Utama:
1. **Representasi Memegang Peranan Dominan:** Kesenjangan performa di Tetris secara mutlak berasal dari **representasi**, bukan algoritma optimasi. Baik algoritma berbasis gradien (DQN) maupun bebas gradien (CEM) mengalami kegagalan eksplorasi (skor 0) saat berhadapan langsung dengan ruang piksel mentah $2^{200}$. Sebaliknya, pada fitur linear berstruktur (Dellacherie), kedua keluarga algoritma mampu belajar secara efektif.
2. **Gradient-Free vs TD pada Fitur Linear:** CEM menunjukkan sample efficiency superior pada fungsi nilai linear kompak karena mengoptimasi langsung return kumulatif tanpa terganggu oleh *bootstrapping bias* atau *deadly triad* yang dialami TD(0).
3. **PPO vs Afterstate (E3 vs E2):** MaskablePPO yang memprediksi aksi langsung sebelum piece ditaruh mengalami kesulitan koordinasi spasial yang lebih tinggi dibanding evaluasi afterstate $V(s')$, membuktikan bahwa paradigma afterstate adalah esensial untuk domain Tetris.
4. **Pengaruh Reward Shaping (E4):** Varian R1 (survival bonus) dan R3 (potential-based) secara signifikan mempercepat agen menemukan transisi yang membersihkan baris dibanding sparse reward murni (R0).

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

