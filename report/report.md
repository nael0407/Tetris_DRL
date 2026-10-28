# Tetris RL: Algoritma atau Representasi? Analisis Komparatif Cross-Entropy Method vs Deep Reinforcement Learning

**Penulis/Peneliti:** Natanael  
**Status Proyek:** Sedang Berjalan (In Progress)  
**Target Environment:** Tetris Berbasis Penempatan Akhir (Gymnasium, CPU-only Linux)  

---

## Abstrak
*(Bagian ini akan disintesis setelah eksperimen utama E1–E5 selesai, menyajikan ringkasan motivasi, rancangan faktorial 2x2, serta temuan empiris utama secara objektif).*

---

## 1. Pendahuluan
Tetris telah lama menjadi salah satu lingkungan tolok ukur (*benchmark*) klasik dalam domain kecerdasan buatan dan *Reinforcement Learning* (RL). Di satu sisi, metode pencarian bebas gradien (*gradient-free*) seperti Cross-Entropy Method (CEM) yang dikombinasikan dengan fitur buatan tangan (*handcrafted linear features*, e.g., fitur Dellacherie) terbukti mampu bertahan jutaan langkah. Di sisi lain, agen *Deep RL* modern yang beroperasi langsung pada representasi papan biner mentah sering kali memerlukan jutaan sampel interaksi namun kerap terjebak pada performa suboptimal jika tidak diberi perlakuan reward shaping yang ekstensif.

Kesenjangan performa ini memunculkan pertanyaan mendasar:
> **Pertanyaan Riset Utama:** Jika terdapat perbedaan performa yang signifikan antara metode klasik berbasis fitur dan Deep RL pada Tetris, apakah perbedaan tersebut berakar dari **algoritma** (*gradient-free optimization* vs *temporal difference/policy gradient*) ataukah dari **representasi keadaan** (*linear handcrafted features* vs *raw spatial board*)?

Proyek ini bertujuan untuk mengisolasi dan menjawab pertanyaan tersebut secara kausal melalui rancangan eksperimen faktorial $2 \times 2$.

---

## 2. Latar Belakang Singkat
- **Tetris sebagai Benchmark RL:** Karakteristik *stochastic transitions*, *delayed reward*, serta risiko keruntuhan kumulatif akibat tumpukan buruk (*cliff-fall dynamics*).
- **Cross-Entropy Method (CEM):** Pendekatan optimasi parameter berbasis populasi untuk estimasi bobot fungsi nilai linear (Szita & Lorincz, 2006).
- **Temporal Difference & Deep Q-Networks (DQN):** Pendekatan estimasi *action-value* berbasis *gradient descent* dan *experience replay*, khususnya dalam konteks *afterstate value evaluation*.
- **Proximal Policy Optimization (PPO):** Metode *policy gradient* on-policy yang dikombinasikan dengan *action masking* (`MaskablePPO`) untuk menangani ruang aksi diskrit dengan kendala aturan legalitas.

---

## 3. Metode
### 3.1 Spesifikasi Game & Environment
- **Ruang Aksi:** *Final Placement* (rotasi + kolom), menghasilkan maksimal 40 aksi diskrit (`Discrete(40)`).
- **Penyederhanaan:** Tidak memodelkan gerakan *tuck* (sliding di bawah overhang), *wall kick*, *T-spin*, waktu jatuh/gravitasi inkremental, maupun fitur *hold*.
- **Sistem Skor Evaluasi:** Standar 1 baris = 100 poin, 2 baris = 300 poin, 3 baris = 500 poin, 4 baris (Tetris) = 800 poin.

### 3.2 Ruang Observasi
Eksperimen membandingkan tiga modalitas observasi:
1. **Raw Board:** Matriks biner $H \times W$ ditambah representasi one-hot piece aktif (7) dan piece berikutnya (7).
2. **Handcrafted Features:** Fitur ringkas meliputi profil tinggi tiap kolom, jumlah lubang (*holes*), kebergelombangan (*bumpiness*), tinggi agregat, dan potensi baris penuh.
3. **Combined:** Kombinasi fitur spasial mentah dan fitur heuristik.

### 3.3 Penanganan Legalitas Aksi & Afterstate
- **Action Masking:** Vektor biner panjang 40 yang mengeliminasi penempatan yang melebihi batas dinding kanan papan serta rotasi duplikat simetris.
- **Afterstate Formulation:** Untuk agen value-based (DQN/CEM), evaluasi dilakukan langsung terhadap keadaan papan deterministik hasil penempatan (*afterstate* $S'_a$).

### 3.4 Desain Faktorial 2x2
Untuk mengurai pengaruh algoritma dan representasi secara terpisah tanpa bias perancangan, eksperimen inti disusun dalam matriks faktorial:

| Algoritma \ Representasi | Fitur Linear Handcrafted | Jaringan Saraf (Papan Mentah) |
|---|---|---|
| **CEM (Gradient-Free)** | Baseline Klasik (Linear Value) | Neuroevolution / Net-CEM |
| **TD / DQN (Gradient-Based)** | Linear TD(0) / Linear DQN | Deep Q-Network Penuh (DQN) |

---

## 4. Protokol Eksperimen
- **Ukuran Papan & Pilot Study:** Menentukan tinggi papan efektif (mis. 10x12, 10x16, atau 10x20) melalui pengujian awal untuk memitigasi risiko *ceiling effect*.
- **Penyetaraan Anggaran Sampel:** Jumlah langkah interaksi dihitung secara setara di semua metode. Untuk CEM, total sampel dihitung dari $\sum (\text{populasi} \times \text{game} \times \text{piece})$.
- **Evaluasi Multi-Seed & Statistik Kuat:** Minimal 5 random seed independen per konfigurasi. Evaluasi dilaporkan menggunakan *Interquartile Mean* (IQM) dan bootstrap confidence interval 95% (`rliable`).
- **Data Tersensor:** Batas evaluasi maksimal 10.000 piece per episode, dianalisis menggunakan estimator survival Kaplan-Meier.

---

## 5. Hasil Eksperimen
*(Bagian ini akan diisi setelah eksperimen E1–E5 dijalankan)*

### 5.1 E1: Baseline Klasik (Random, Heuristik, CEM Linear)
### 5.2 E2: Analisis Faktorial 2x2 (Algoritma vs Representasi)
### 5.3 E3: Evaluasi Policy Gradient (MaskablePPO) & Observasi Gabungan
### 5.4 E4: Pengaruh Skema Reward Shaping (R0 – R3)
### 5.5 E5: Efisiensi Afterstate vs Pemilihan Aksi Langsung

---

## 6. Interpretabilitas
*(Bagian ini akan diisi setelah pelatihan agen selesai)*
- **Linear Probing Value Function:** Analisis korelasi representasi representasi lapisan tersembunyi jaringan terhadap fitur human-interpretable (lubang, kontur, tinggi).
- **Eksperimen Intervensi Papan:** Pengujian kausal respon nilai terhadap penambahan lubang artifisial dan tumpukan ekstrim.

---

## 7. Analisis Kegagalan
*(Dokumentasi pola kegagalan, skenario bottleneck, dan pola piece berbahaya seperti distribusi S/Z berturut-turut).*

---

## 8. Diskusi dan Keterbatasan
- Implikasi simplifikasi *final placement* tanpa *tuck*.
- Generalisasi temuan ke ranah representasi spasial game diskrit.

---

## 9. Kesimpulan
*(Jawaban atas pertanyaan riset utama berdasarkan bukti empiris).*

---

## 10. Referensi
1. Szita, I., & Lőrincz, A. (2006). *Learning Tetris using the noisy cross-entropy method*. Neural computation, 18(12), 2936-2941.
2. Gabillon, V., Ghavamzadeh, M., & Scherrer, B. (2013). *Approximate dynamic programming finally performs well in the game of Tetris*. Advances in Neural Information Processing Systems (NeurIPS).
3. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. (2021). *Deep reinforcement learning at the edge of the statistical precipice*. Advances in Neural Information Processing Systems (NeurIPS).
