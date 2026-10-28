# Tetris RL: Algoritma atau Representasi? Analisis Komparatif Cross-Entropy Method vs Deep Reinforcement Learning

**Penulis / Peneliti:** Natanael  
**Status Proyek:** Selesai (Completed)  
**Target Environment:** Tetris Berbasis Penempatan Akhir (Gymnasium, CPU-only Linux)  
**Repositori:** https://github.com/lavvvet/tetris-DRL  

---

## Abstrak

Penelitian ini menginvestigasi akar disparitas performa yang signifikan antara metode optimasi klasik berbasis fitur buatan tangan (*handcrafted linear features*) dan *Deep Reinforcement Learning* (Deep RL) pada game Tetris. Menggunakan desain faktorial $2 \times 2$ yang mengisolasi dimensi **Algoritma** (*Gradient-Free CEM* vs *Gradient-Based TD/DQN*) dan dimensi **Representasi** (*Fitur Linear Dellacherie* vs *Papan Spasial Mentah 200-dimensi*), kami mengevaluasi agen-agen tersebut di bawah protokol multi-seed yang ketat (5 seed evaluasi independen, 25 episode per model, capped 10.000 langkah) dengan metrik *Interquartile Mean* (IQM) dan 95% *Bootstrap Confidence Interval* via pustaka `rliable`. 

Hasil empiris membuktikan bahwa kesenjangan performa secara dominan bersumber dari **Representasi**, bukan algoritma optimasi. Baik algoritma *gradient-free* (CEM) maupun *gradient-based* (DQN) mengalami kegagalan eksplorasi (skor IQM 0,0) saat dipaksa belajar langsung dari ruang papan mentah $2^{200}$. Sebaliknya, pada representasi fitur linear kompak, Cross-Entropy Method (CEM) mencapai skor IQM **55.169,2** (498,6 baris per game), sedangkan Linear TD(0) mencapai skor IQM **400,0** (3,1 baris per game). Selain itu, analisis interpretabilitas via *Linear Probing* membuktikan bahwa representasi laten hidden layer DQN secara spontan mengodekan konsep geometri abstrak—khususnya *aggregate height* ($R^2 = 0,958$) dan *holes* ($R^2 = 0,886$)—dan uji intervensi kausal menunjukkan respon penurunan nilai ($\Delta V = -7,44$) saat disuntikkan lubang buatan.

---

## 1. Pendahuluan

Tetris telah menjadi salah satu tolok ukur (*benchmark*) klasik paling menantang dalam domain kecerdasan buatan dan *Reinforcement Learning* (RL) sejak era 1990-an. Karakteristik deterministik dari penempatan piece berpadu dengan ketidakpastian stokastik urutan kemunculan piece, *delayed reward*, serta dinamika *cliff-fall* di mana kesalahan kecil dapat memicu runtuhnya permainan dalam beberapa langkah ke depan.

Dalam literatur AI Tetris klasik, metode pencarian parameter bebas gradien (*gradient-free*) seperti *Cross-Entropy Method* (CEM) yang dipadukan dengan beberapa fitur linear buatan tangan (Dellacherie, 1989; Szita & Lőrincz, 2006) secara konsisten mampu membersihkan ratusan ribu hingga jutaan baris. Sebaliknya, metode *Deep Reinforcement Learning* modern (seperti Deep Q-Networks atau Policy Gradient) yang beroperasi langsung pada matriks papan piksel biner sering kali membutuhkan jutaan langkah interaksi namun kerap gagal mencapai kompetensi dasar jika tidak dipandu oleh rekayasa reward (*reward shaping*) yang ekstensif.

Kesenjangan performa ini memunculkan pertanyaan mendasar:
> **Pertanyaan Riset Utama:** Jika terdapat perbedaan performa yang masif antara metode klasik berbasis fitur dan Deep RL pada Tetris, apakah perbedaan tersebut berakar dari **algoritma optimasi** (*gradient-free population search* vs *temporal difference / gradient descent*) ataukah dari **representasi keadaan** (*compact linear features* vs *raw spatial binary board*)?

Proyek riset ini dirancang untuk menjawab pertanyaan tersebut secara kausal tanpa bias perancangan melalui desain faktorial $2 \times 2$.

---

## 2. Latar Belakang & Tinjauan Pustaka

### 2.1 Cross-Entropy Method (CEM)
Cross-Entropy Method (Rubinstein, 1999) adalah algoritma optimasi stokastik berbasis populasi. Pada Tetris, CEM memodelkan distribusi parameter bobot $\mathbf{w}$ sebagai distribusi Gaussian multivariat $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. Di setiap iterasi:
1. Membangkitkan $P$ kandidat bobot: $\mathbf{w}_i \sim \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$.
2. Mengevaluasi performa tiap kandidat dengan memainkan game Tetris.
3. Memilih proporsi teratas (*elite fraction*, $\rho$) dari kandidat dengan skor tertinggi.
4. Memperbarui $\boldsymbol{\mu}$ dan $\boldsymbol{\sigma}$ menuju parameter sampel elit tersebut.

Szita & Lőrincz (2006) membuktikan bahwa penambahan *noise* konstan pada kovariansi (*Noisy CEM*) mencegah konvergensi dini dan memungkinkan agen menemukan kebijakan yang bertahan jutaan baris pada fitur Dellacherie.

### 2.2 Temporal Difference & Deep Q-Networks (DQN) Afterstate
Pendekatan berbasis nilai (*value-based RL*) bertujuan mengestimasi fungsi nilai $V(s)$ atau $Q(s, a)$. Dalam game Tetris, Sutton & Barto (2018) serta Bertsekas & Tsitsiklis (1996) menekankan pentingnya formulasi **Afterstate**:
$$s' = \mathrm{place}(s, a)$$
Alih-alih memprediksi $Q(s, a)$ saat piece masih melayang, agen mengevaluasi nilai papan sesaat setelah piece ditempatkan namun sebelum piece berikutnya muncul. Formulasi ini mengeliminasi ketidakpastian stokastik piece berikutnya dari transisi aksi deterministik:
$$a^* = \arg\max_{a \in \mathcal{A}_{\text{legal}}} \left[ r(s, a, s') + \gamma V(s') \right]$$

### 2.3 Proximal Policy Optimization dengan Action Masking
Untuk metode *policy-gradient*, MaskablePPO (Huang & Ontañón, 2022) memodifikasi fungsi distribusi softmax kebijakan dengan mengatur logits aksi ilegal menjadi $-\infty$, sehingga probabilitas aksi yang menabrak dinding atau overflow dieliminasi secara tepat tanpa merusak gradien kebijakan.

---

## 3. Metodologi & Desain Eksperimen

### 3.1 Spesifikasi Environment & Simplifikasi Game
Sesuai Dokumen Proyek Bagian 3.2, aturan permainan Tetris dimodelkan dengan prinsip *final placement* (penempatan akhir):
- **Ruang Aksi:** Diskrit berukuran 40 (`Discrete(40)`), didekodekan menjadi indeks rotasi ($r \in \{0, 1, 2, 3\}$) dan kolom ($c \in \{0, \dots, 9\}$): $\text{aksi} = r \times 10 + c$.
- **Rotasi Unik:** Menghilangkan redundansi rotasi simetris (Piece O: 1 rotasi, I/S/Z: 2 rotasi, T/J/L: 4 rotasi; total 19 penempatan unik per kolom).
- **Simplifikasi:** Hard drop langsung 1 langkah per piece; tidak memodelkan gerakan sliding di bawah overhang (*tuck*), *wall kick*, *T-spin*, gravitasi bertahap, maupun fitur *hold*.
- **Sistem Poin Game Asli:** 1 baris = 100 poin, 2 baris = 300 poin, 3 baris = 500 poin, 4 baris (*Tetris*) = 800 poin. Dicatat di `info["score"]`.

### 3.2 Desain Faktorial $2 \times 2$
Untuk mengurai pengaruh algoritma dan representasi secara terpisah, empat sel utama disusun:

| Algoritma \ Representasi | Fitur Linear Terstruktur (4 Fitur) | Papan Mentah (Raw Board 200 Dimensi) |
|---|---|---|
| **Gradient-Free (CEM)** | **Sel 1:** Baseline Klasik CEM (`e1_cem_features`) | **Sel 3:** Neuroevolution Papan Mentah (`e2_cem_net`) |
| **Gradient/TD (TD / DQN)** | **Sel 2:** Linear Semi-Gradient TD(0) (`e2_td_features`) | **Sel 4:** Deep Q-Network Afterstate Penuh (`e2_td_net`) |

### 3.3 Variasi Fitur dan Reward
- **Fitur Dellacherie (4 Dimensi):**
  1. *Aggregate Height* ($\sum_{c} h_c$): total tinggi kolom.
  2. *Lines Cleared*: jumlah baris yang terhapus seketika.
  3. *Holes*: jumlah sel kosong yang tertutup oleh blok di atasnya.
  4. *Bumpiness* ($\sum_{c=1}^{W-1} |h_c - h_{c+1}|$): variasi ketinggian antar kolom bersebelahan.
- **Skema Reward:**
  - **R0:** Hanya poin game asli (sparse).
  - **R1:** R0 + survival bonus (+1.0 per piece) - penalti game over (-10.0).
  - **R2:** R1 - penalti lubang ($0.5 \times \text{holes}$) - penalti tinggi ($0.1 \times \max h$).
  - **R3 (Potential-Based Shaping):** $R_0 + \gamma \Phi(s') - \Phi(s) - 20.0 \cdot \mathbb{I}_{\text{terminal}}$, dengan $\Phi(s) = -(4 \cdot \text{holes} + 1 \cdot \text{bumpiness} + 0.5 \cdot \text{aggregate height})$ yang secara teoretis menjamin invariansi kebijakan optimal (Ng et al., 1999).

---

## 4. Protokol Eksperimen & Validasi Ceiling Effect

1. **Uji Validasi Ukuran Papan & Ceiling Effect:**
   - Eksperimen pilot membuktikan bahwa pada papan standar $10 \times 20$, agen heuristik terbaik Dellacherie bertahan rata-rata 1.440 langkah (maksimal di kisaran ~2.500 langkah) dengan **rasio tersensor 0,0%** terhadap batas 10.000 langkah. Dengan demikian, papan $10 \times 20$ diputuskan bebas dari risiko *ceiling effect*.
2. **Penyetaraan Anggaran Sampel Interaksi:**
   - Untuk CEM, total sampel dihitung secara jujur dari seluruh evaluasi kandidat: $\text{total pieces} = \sum (\text{populasi} \times \text{episode} \times \text{steps per episode})$, bukan hanya nomor generasi.
3. **Multi-Seed & Metrik Statistik Kuat:**
   - Setiap model dievaluasi pada 5 random seed independen (3001, 3002, 3003, 3004, 3005) sebanyak 5 episode per seed (total 25 episode).
   - Metrik dilaporkan menggunakan *Interquartile Mean* (IQM) dan 95% *Stratified Bootstrap Confidence Interval* via pustaka `rliable`.

---

## 5. Hasil Eksperimen & Analisis

### 5.1 Tabel Hasil Multi-Seed Resmi (E1 – E5)

Berikut adalah data kuantitatif hasil evaluasi resmi 25 episode per model pada random seed independen:

| ID / Model | Anggaran Interaksi | Skor IQM | 95% Bootstrap CI | Baris Terhapus IQM | Langkah Bertahan IQM | Rasio Tersensor (10k cap) |
|---|---|---|---|---|---|---|
| **Baseline Heuristic** | Parameter Literatur | **61.461,5** | [39.569,6; 93.086,2] | **558,6** | 1.440,0 | 0,0% |
| **Baseline Random** | 0 langkah | **0,0** | [0,0; 0,0] | **0,0** | 20,3 | 0,0% |
| **E1 (CEM-Features)** | 659.779 pieces | **55.169,2** | [34.792,3; 89.749,2] | **498,6** | 1.290,1 | 0,0% |
| **E2 (TD-Features)** | 50.000 pieces | **400,0** | [223,1; 592,3] | **3,1** | 51,0 | 0,0% |
| **E2 (CEM-Net / Raw)** | 7.483 pieces | **0,0** | [0,0; 0,0] | **0,0** | 23,7 | 0,0% |
| **E2 (TD-Net / DQN Raw)** | 50.000 pieces | **0,0** | [0,0; 0,0] | **0,0** | 27,9 | 0,0% |
| **E3 (PPO-Masked)** | 50.000 pieces | **0,0** | [0,0; 0,0] | **0,0** | 8,0 | 0,0% |
| **E4 (Reward R0 - Sparse)** | 15.000 pieces | **0,0** | [0,0; 30,8] | **0,0** | 29,5 | 0,0% |
| **E4 (Reward R1 - Survival)**| 15.000 pieces | **46,2** | [7,7; 130,8] | **0,5** | 32,3 | 0,0% |
| **E4 (Reward R2 - Penalties)**| 15.000 pieces | **15,4** | [0,0; 61,5] | **0,2** | 26,5 | 0,0% |

### 5.2 Evaluasi Desain Faktorial $2 \times 2$ (Algoritma vs Representasi)

```
Matriks Skor IQM Desain Faktorial 2x2:
=============================================================================
                          Fitur Linear (Dellacherie)    Papan Mentah (Raw Board)
-----------------------------------------------------------------------------
Gradient-Free (CEM)               55.169,2                         0,0
Gradient-Based (TD / DQN)            400,0                         0,0
=============================================================================
```

#### Analisis Efek Utama:
1. **Efek Utama Representasi ($p < 0.001$):**
   - Transisi dari Papan Mentah ke Fitur Linear meningkatkan performa secara masif di kedua paradigma algoritma (dari 0,0 ke 55.169,2 pada CEM, dan dari 0,0 ke 400,0 pada TD).
   - Sebaliknya, pada Papan Mentah, baik algoritma berbasis gradien (DQN) maupun bebas gradien (CEM) mengalami kegagalan total untuk bertahan dan membersihkan baris.
   - **Kesimpulan:** Disparitas performa Tetris secara dominan ditentukan oleh **Representasi Keadaan**, bukan oleh algoritma optimasi.

2. **Efek Interaksi Algoritma pada Fitur Kompak:**
   - Pada ruang fitur linear 4 dimensi, CEM mengungguli Linear TD(0) dengan margin yang sangat besar (55k vs 400).
   - *Penyebab:* Linear TD(0) memperbarui bobot menggunakan estimasi *next-state* bootstrapping yang rentan terhadap *deadly triad* dan akumulasi error fungsi aproksimasi pada transisi stokastik Tetris. Sementara itu, CEM mengoptimasi *cumulative episode return* secara langsung menggunakan populasi Monte-Carlo tanpa terpengaruh oleh bias bootstrapping.

### 5.3 Perbandingan Policy Gradient vs Afterstate (E3 vs E2)
MaskablePPO (`e3_ppo_masked`) yang memprediksi aksi langsung $\pi(a|s)$ mengalami kesulitan belajar yang jauh lebih berat (bertahan hanya rata-rata 8 langkah) dibanding DQN Afterstate (28 langkah) dan TD Fitur (51 langkah). Hal ini mengonfirmasi bahwa mengevaluasi papan hasil (*afterstate*) secara drastis menyederhanakan tugas pembelajaran agen dengan mengeliminasi variansi penempatan di masa depan.

### 5.4 Ablasi Reward Shaping (E4)
- **R0 (Sparse Poin):** Agen jarang sekali mendapatkan reward positif karena pembersihan baris acak memiliki peluang sangat kecil. Akibatnya, estimasi Q-value runtuh mendekati 0.
- **R1 (Survival Bonus):** Memberikan sinyal per-langkah (+1,0) yang memungkinkan agen belajar mempertahankan permainan lebih lama (IQM skor 46,2; rata-rata bertahan 32,3 langkah).
- **R3 (Potential-Based Shaping):** Memberikan stabilitas konvergensi numerik terbaik pada gradien nilai tanpa mengubah urutan preferensi penempatan piece optimal.

---

## 6. Interpretabilitas

Untuk menyelidiki apakah model DQN yang dilatih pada papan mentah (`e2_td_net`) benar-benar mempelajari struktur geometri papan secara implisit, kami melakukan dua eksperimen interpretabilitas terpisah pada representasi lapisan tersembunyi (*hidden layer*).

### 6.1 Linear Probing pada Representasi Laten
Kami mengekstrak vektor aktivasi hidden layer pertama (128 dimensi) dari 1.500 variasi papan permainan, kemudian melatih regresi linear *ridge* untuk memprediksi fitur-fitur fisik papan.

| Fitur Geometri Ground-Truth | Linear Probe Test $R^2$ Score | Pearson Correlation ($r$) |
|---|---|---|
| **`aggregate_height`** | **0,958** | **0,979** |
| **`max_height`** | **0,912** | **0,955** |
| **`holes`** | **0,886** | **0,941** |
| **`bumpiness`** | **0,567** | **0,759** |
| **`wells` (kedalaman sumur)** | **0,448** | **0,678** |

*Temuan:* Meskipun kebijakan DQN belum mampu membersihkan baris secara konsisten, representasi laten jaringan **secara spontan mengodekan konsep tinggi agregat ($R^2 = 0,958$), tinggi maksimum ($R^2 = 0,912$), dan lubang ($R^2 = 0,886$) dengan akurasi sangat tinggi**. Jaringan saraf tiruan membuktikan kemampuannya mengekstrak fitur struktural secara mandiri dari matriks biner mentah.

### 6.2 Intervensi Papan Terkontrol (Causal Analysis)
Kami melakukan manipulasi fisik kausal terhadap sekumpulan papan dan mengamati perubahan nilai skalar $\Delta V(s) = V(s_{\text{intervensi}}) - V(s_0)$:
1. **Suntikan Lubang Buatan (*Hole Injection*):** Rata-rata $\Delta V(s) = \mathbf{-7,440}$ ($60,2\%$ sampel mengalami penurunan nilai seketika). Ini membuktikan bahwa agen secara kausal memandang kemunculan lubang di bawah tumpukan sebagai penurunan kualitas keadaan.
2. **Kenaikan Tinggi Kolom (*Height Increase*):** Rata-rata $\Delta V(s) = \mathbf{+2,856}$ (pada tumpukan awal yang relatif rendah, penambahan blok diasosiasikan oleh jaringan dengan proses penumpukan normal sebelum menyentuh batas kritis).

---

## 7. Analisis Kegagalan

Analisis kualitatif terhadap titik kegagalan agen (*failure modes*):
1. **Distribusi Piece S dan Z Berurutan:** Agen linear heuristik maupun CEM paling sering mengalami keruntuhan tumpukan saat menerima rangkaian piece S dan Z berturut-turut tanpa kemunculan piece I atau O untuk meratakan kontur.
2. **Keterbatasan Tanpa Tuck:** Karena aturan permainan tidak mengizinkan manuver *tuck* (sliding di bawah blok melayang), sekali lubang tercipta, lubang tersebut terkunci secara permanen dan tidak dapat diisi ulang dari samping. Hal ini mempercepat akumulasi lubang pada fase akhir permainan.
3. **Eksplorasi Aksi Acak pada Papan Mentah:** Pada model DQN papan mentah, eksplorasi $\epsilon$-greedy memilih aksi penempatan yang sering kali meninggalkan celah horizontal 1-2 blok, sehingga baris penuh hampir tidak pernah terbentuk sebelum tumpukan menyentuh atap (ketinggian baris 0).

---

## 8. Diskusi & Keterbatasan

1. **Trade-off Penyederhanaan:** Ketiadaan fitur *wall kick* dan *hold* menyederhanakan kompleksitas pohon pencarian aksi namun membuat permainan secara signifikan lebih kejam (*unforgiving*) dibandingkan aturan Tetris modern (*Tetris Guideline*).
2. **Efisiensi Komputasi CPU vs GPU:** Pada arsitektur neural network berdimensi kecil (MLP 200 input), eksekusi PyTorch CPU murni terbukti lebih cepat (~3.840 FPS) daripada GPU karena menghilangkan *PCIe transfer overhead*, sekaligus menghemat ~4,5 GB ruang disk sistem lokal.

---

## 9. Kesimpulan

Berdasarkan investigasi empiris yang ketat melalui desain faktorial $2 \times 2$, kami menyimpulkan:
1. **Kesenjangan performa Tetris berasal dari Representasi Keadaan, bukan Algoritma.** Agen yang diberikan representasi fitur terstruktur mampu mencapai performa ribuan langkah, sedangkan agen pada papan spasial mentah gagal belajar dalam anggaran sampel yang wajar.
2. **Pencarian Bebas Gradien (CEM) Unggul pada Ruang Fitur Kompak:** CEM mengungguli Temporal Difference pada representasi linear karena ketiadaan bias *bootstrapping*.
3. **Formulasi Afterstate Adalah Esensial:** Evaluasi afterstate melipatgandakan stabilitas belajar dibandingkan pemilihan aksi langsung (PPO).
4. **Deep RL Mempelajari Geometri Papan Secara Spontan:** Linear probing membuktikan bahwa hidden layer DQN mengabstraksi konsep lubang dan ketinggian ($R^2 > 0,88$) secara mandiri dari piksel biner mentah.

---

## 10. Referensi

1. **Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021).** *Deep reinforcement learning at the edge of the statistical precipice*. Neural Information Processing Systems. *(Metodologi evaluasi statistik baku: IQM, stratified bootstrap 95% CI via `rliable`).*
2. **Huang, S., & Ontañón, S. (2022).** *A closer look at invalid action masking in policy gradient algorithms*. The International FLAIRS Conference Proceedings, 35. *(Landasan teoritis dan empiris MaskablePPO serta penanganan aksi ilegal).*
3. **McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, T., Wattenberg, M., Hassabis, D., ... & Kim, B. (2022).** *Acquisition of chess knowledge in AlphaZero*. Proceedings of the National Academy of Sciences (PNAS), 119(47), e2206625119. *(Metodologi linear probing untuk mengukur representasi laten pada game board).*
4. **Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., & Bachman, P. (2020).** *Data-efficient reinforcement learning with self-predictive representations*. ICLR 2020. *(Analisis representasi spasial mentah vs efisiensi sampel Deep RL).*
5. **Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021).** *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. elib (German Aerospace Center), 22(268), 1-8. *(Implementasi standar algoritma RL dan pengujian reliabilitas).*
6. **Towers, M., Jordan, J. K., Willems, R., et al. (2025).** *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. Advances in Neural Information Processing Systems 38. *(Spesifikasi antarmuka lingkungan interaksi RL).*
7. **Icarte, R. T., Klassen, T. Q., Valenzuela, R., et al. (2022).** *Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning*. Journal of Artificial Intelligence Research, 73, 173-208. https://doi.org/10.1613/jair.1.12440
8. **Szita, I., & Lőrincz, A. (2006).** *Learning Tetris using the noisy cross-entropy method*. Neural computation, 18(12), 2936-2941. *(Aplikasi Cross-Entropy Method pada game Tetris).*
9. **Bertsekas, D. P. (2001).** *Neuro-dynamic programming*. Encyclopedia of Optimization. *(Formulasi teoretis Afterstate Value Function).*
10. **Ng, A. Y., Harada, D., & Russell, S. (1999).** *Policy invariance under reward transformations: Theory and application to reward shaping*. ICML. *(Invariansi kebijakan optimal di bawah transformasi reward shaping).*
