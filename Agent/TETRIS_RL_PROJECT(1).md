# Tetris RL: Algoritma atau Representasi? CEM vs Deep RL di Tetris

Dokumen perencanaan proyek portofolio. Dikerjakan lokal di Linux (Ubuntu), Python, tanpa kebutuhan GPU.

---

## 1. Ringkasan

Proyek ini membangun environment Tetris sendiri (penempatan akhir), melatih agen deep RL, lalu membandingkannya dengan metode klasik berbasis fitur buatan tangan. Fokusnya bukan membuat agen yang jago, tetapi **menjawab pertanyaan riset dengan eksperimen yang rapi**, lalu membedah apa yang dipelajari agen dan menampilkannya lewat visualisasi.

**Pertanyaan riset utama (arah A):**
Jika ada selisih performa antara metode klasik berbasis fitur dan deep RL di Tetris, apakah selisih itu berasal dari **algoritma** (gradient-free vs TD/gradient) atau dari **representasi** (fitur buatan tangan vs papan mentah)? Dijawab lewat desain faktorial 2x2 (lihat 5.4), dengan reward shaping sebagai faktor tambahan. Hasilnya bisa saja menunjukkan bahwa tidak ada selisih berarti.

**Arah tambahan:**
- **C. Interpretabilitas:** apa yang "dilihat" dan dipelajari agen terlatih.
- **E. Demo:** viewer langsung, rekaman per checkpoint, dan mode manusia vs agen.
- **B. Reward shaping:** menjadi variabel di dalam eksperimen A, bukan arah sendiri.
- **D. Generalisasi ke varian aturan:** ditunda, dikerjakan jika waktu cukup.

---

## 2. Keputusan Desain (sudah disepakati)

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Ruang aksi | Penempatan akhir (rotasi + kolom) | Ruang aksi kecil (maks 40), latihan cepat dan stabil, adil dibanding heuristik klasik |
| Environment | Ditulis sendiri (Python + NumPy + Gymnasium) | Kontrol penuh atas aturan, observasi, reward, seed |
| Papan | 10x20 sebagai target awal, tinggi final ditentukan lewat pilot (lihat 3.5) | 10x20 standar literatur, tetapi berisiko *ceiling effect* |
| Pengacak piece | Acak seragam | Lebih sulit, lebih mudah dibandingkan dengan literatur. 7-bag disimpan sebagai variasi |
| Preview | 1 piece berikutnya | Mendekati setting literatur |
| Hold | Tidak ada | Menyederhanakan |
| Garbage lines | Tidak ada | Menyederhanakan |
| Batas episode evaluasi | 10.000 piece, dilaporkan sebagai data tersensor | Agen bagus bisa menyentuh batas, lihat 3.5 |
| Lingkungan kerja | Lokal, Linux Ubuntu, CPU | Viewer langsung dan struktur repo rapi |

---

## 3. Aturan Game

### 3.1 Aturan inti (tetap di semua eksperimen)
- 7 tetromino: I, O, T, S, Z, J, L, masing-masing dengan rotasinya (O: 1, I/S/Z: 2, T/J/L: 4 rotasi unik).
- Piece muncul di atas. Agen memilih (rotasi, kolom), lalu piece jatuh lurus ke bawah sampai menumpu (hard drop).
- Baris penuh dihapus, baris di atasnya turun. Bisa 1 sampai 4 baris sekaligus.
- Game over jika piece baru tidak bisa muncul karena area spawn terisi.
- Semua keacakan dikontrol seed (reproduktibel).

### 3.2 Simplifikasi yang WAJIB dicatat di laporan
Dengan penempatan akhir, hal berikut **tidak dimodelkan**: gerakan *tuck* (menggeser piece ke bawah overhang), T-spin, *wall kick*, kecepatan jatuh/gravitasi, dan hold. Ini umum di riset Tetris AI, tetapi harus disebut eksplisit agar hasil tidak dikira mewakili Tetris penuh.

### 3.3 Skor (metrik evaluasi, tetap)

| Baris dihapus | Poin |
|---|---|
| 1 | 100 |
| 2 | 300 |
| 3 | 500 |
| 4 (Tetris) | 800 |

**Skor game** dipakai untuk evaluasi dan tidak berubah. **Reward pelatihan** adalah hal terpisah dan menjadi variabel eksperimen.

### 3.4 Parameter variasi (untuk eksperimen generalisasi nanti)

| Parameter | Mudah | Standar | Sulit |
|---|---|---|---|
| Preview | 3 piece | 1 piece | 0 piece |
| Hold | Ada | Tidak | Tidak |
| Pengacak | 7-bag | Acak seragam | Bias ke S/Z |
| Tinggi papan | 24 | 20 | 12 |
| Garbage | Tidak | Tidak | Berkala |

Semua parameter ini cukup argumen konstruktor environment, tanpa mengubah kode inti.

### 3.5 Risiko ceiling effect (batas 10.000 piece)

Pada penempatan akhir tanpa tuck, metode klasik yang baik (CEM + fitur Dellacherie) kemungkinan besar bermain sangat lama di papan 10x20. Jika semua baseline dan agen bagus sama-sama berhenti di batas 10.000 piece, E1 dan E2 tidak bisa membedakan mereka (semua "tamat di batas").

Rencana:
1. **Pilot dulu, jangan ditebak.** Setelah baseline CEM/heuristik jalan (tahap 6), jalankan pada beberapa tinggi papan (mis. 10x10, 10x12, 10x16, 10x20) dan lihat berapa persen run menyentuh batas.
2. **Pilih tinggi papan utama** yang membuat game masih bisa kalah untuk agen yang baik. 10x20 tetap dilaporkan sebagai pembanding tersensor jika memungkinkan.
3. **Laporkan sebagai data tersensor:** proporsi run yang mencapai batas, plus kurva survival (Kaplan-Meier), bukan hanya rata-rata.
4. Konsekuensi: papan lebih kecil mengurangi keterbandingan langsung dengan angka di literatur (yang umumnya 10x20). Ini trade-off yang dicatat di laporan.

---

## 4. Bagaimana Deep RL Mengakses Tetris (antarmuka agen-environment)

Agen RL **tidak melihat layar**. Ia berinteraksi lewat antarmuka standar **Gymnasium**: environment adalah objek Python dengan dua metode utama.

```python
obs, info = env.reset(seed=42)                    # mulai episode baru
obs, reward, terminated, truncated, info = env.step(action)   # ambil satu aksi
```

Library RL (Stable-Baselines3, CleanRL, atau kode PyTorch sendiri) hanya memanggil dua metode ini dalam loop. Selama environment mengikuti antarmuka Gymnasium, algoritma apa pun bisa dipasang tanpa mengubah game.

### 4.1 Alur satu langkah (step)
1. Agen menerima `obs` (keadaan papan + piece sekarang + piece berikutnya).
2. Agen memilih `action` (bilangan bulat).
3. Environment mendekode aksi menjadi (rotasi, kolom), menjatuhkan piece, menghapus baris, dan memunculkan piece baru.
4. Environment mengembalikan observasi baru, reward, dan penanda selesai.

### 4.2 Ruang aksi
`Discrete(40)` = 4 rotasi x 10 kolom. Aksi `a` didekode:

```
rotasi = a // 10
kolom  = a % 10
```

Tidak semua aksi valid, dan ada dua sumber ketidakvalidan:
1. **Keluar papan:** mis. I horizontal (lebar 4) hanya valid mulai kolom 0 sampai 6.
2. **Rotasi duplikat:** O hanya punya 1 rotasi unik, I/S/Z punya 2, T/J/L punya 4. Indeks rotasi di atas jumlah rotasi unik harus ikut di-mask. Tanpa ini, policy melihat aksi berbeda nomor yang hasilnya identik.

Ada dua cara menanganinya:

| Cara | Penjelasan | Catatan |
|---|---|---|
| **Action masking** | Environment memberi daftar aksi valid (`info["action_mask"]`). Agen hanya boleh memilih yang valid. Dipakai `MaskablePPO` dari `sb3-contrib` | Disarankan untuk PPO |
| **Afterstate (evaluasi per penempatan)** | Untuk setiap penempatan valid, environment menghitung papan hasilnya. Jaringan menilai tiap papan hasil, agen memilih yang nilainya tertinggi | Pendekatan klasik dan efektif untuk Tetris. Cocok untuk DQN gaya value-based |

Rencana: implementasikan **keduanya** dan jadikan perbandingan eksperimen. Environment menyediakan fungsi `valid_placements()` yang mengembalikan daftar (aksi, papan hasil, baris dihapus).

### 4.3 Observasi

Opsi (dibandingkan sebagai eksperimen):

| Opsi | Isi | Bentuk |
|---|---|---|
| **Papan mentah** | Papan biner 20x10 + one-hot piece sekarang (7) + one-hot piece berikutnya (7) | CNN atau MLP |
| **Fitur buatan tangan** | Tinggi tiap kolom, jumlah lubang, kebergelombangan (bumpiness), total tinggi, baris penuh | MLP kecil |
| **Gabungan** | Papan mentah + fitur | CNN + MLP |

### 4.4 Reward pelatihan (variabel eksperimen)

| Varian | Isi |
|---|---|
| R0 | Hanya skor baris dihapus |
| R1 | R0 + bonus kecil per piece bertahan |
| R2 | R1 - penalti lubang - penalti tinggi tumpukan |
| R3 | Perubahan fitur (potential-based shaping) |

Catatan: reward shaping harus dilaporkan dengan jelas dan **evaluasi tetap memakai skor game**, supaya perbandingan adil.

### 4.5 Penanda selesai
- `terminated = True` saat game over.
- `truncated = True` saat mencapai batas piece (10.000 pada evaluasi, nilai lebih kecil saat pelatihan agar episode tidak terlalu panjang).

### 4.6 Wrapper yang akan disediakan
- `TetrisEnv` (Gymnasium) dengan argumen: pengacak, jumlah preview, tinggi papan, jenis observasi, jenis reward.
- Cek kepatuhan dengan `gymnasium.utils.env_checker.check_env`.
- `info` berisi: skor game, baris dihapus, jumlah lubang, tinggi maksimum, dan `action_mask`.

---

## 5. Metode yang Dibandingkan

### 5.1 Baseline (wajib ada)
1. **Agen acak** (batas bawah).
2. **Heuristik berbobot**: pilih penempatan dengan skor linear dari fitur (tinggi agregat, lubang, bumpiness, baris dihapus). Bobot awal dari literatur lalu dioptimasi.
3. **Cross-Entropy Method (CEM)** pada bobot fitur (kuat di Tetris dan menjadi pembanding yang kredibel).

### 5.2 Agen deep RL
1. **DQN/Double DQN dengan afterstate** (value-based).
2. **PPO dengan action masking** (policy-gradient).
3. Variasi: observasi papan mentah vs fitur vs gabungan, dan reward R0 sampai R3.

### 5.3 Prinsip perbandingan yang adil
- Jumlah langkah interaksi sama antar metode. **Untuk CEM, langkah interaksi = total piece yang dimainkan dari semua evaluasi kandidat** (populasi x jumlah game x piece per game), bukan jumlah generasi.
- Minimal **5 seed** per konfigurasi, dengan banyak episode evaluasi per seed. Laporkan interquartile mean (IQM) dan interval kepercayaan bootstrap (pustaka `rliable`), bukan hanya rata-rata +/- simpangan.
- Seed evaluasi berbeda dari seed pelatihan.
- Ukuran jaringan dan anggaran hyperparameter tuning dicatat.

### 5.4 Desain faktorial 2x2 (inti kontribusi)

CEM pada fitur linear pada dasarnya adalah value function linear di afterstate yang dilatih tanpa gradien. Membandingkan "CEM (fitur)" langsung dengan "DQN (papan mentah)" mengubah dua hal sekaligus (algoritma dan representasi), sehingga hasilnya ambigu. Karena itu perbandingan utama dibuat faktorial:

| | Fitur linear | Jaringan di papan mentah |
|---|---|---|
| **CEM (gradient-free)** | Baseline klasik | Neuroevolution |
| **TD/DQN afterstate (gradient)** | TD + fitur linear | DQN penuh |

Dengan ini pertanyaan "algoritma atau representasi yang jadi masalah?" bisa dijawab langsung dari efek utama dan interaksinya. PPO dengan action masking menjadi pembanding tambahan di luar faktorial.

---

## 6. Rencana Eksperimen

| ID | Pertanyaan | Variabel | Metrik |
|---|---|---|---|
| E1 | Seberapa kuat baseline klasik? | Acak, heuristik, CEM | Baris dihapus, skor, lama bertahan |
| E2 | Algoritma atau representasi? | Faktorial 2x2 (lihat 5.4): CEM/TD x fitur linear/jaringan papan mentah | Sama seperti E1, plus sample efficiency |
| E3 | Bagaimana PPO + masking dibanding sel faktorial? Apakah gabungan papan + fitur membantu? | PPO masking; observasi papan mentah vs fitur vs gabungan | Kurva belajar, performa akhir |
| E4 | Seberapa penting reward shaping? | R0 sampai R3 | Performa akhir, stabilitas, perilaku tak diinginkan |
| E5 | Afterstate vs aksi langsung | DQN afterstate vs PPO masking | Performa dan kecepatan belajar |
| E6 (opsional) | Per tombol vs penempatan akhir | Ruang aksi | Kesulitan bertambah berapa |
| E7 (opsional, arah D) | Generalisasi ke varian aturan | Latih standar, uji varian | Generalization gap |

Hipotesis awal (diuji, bukan diasumsikan; hasil bisa saja menunjukkan DQN afterstate + fitur menyamai CEM): jika ada kesenjangan, sebagian besar berasal dari representasi, bukan dari algoritma.

---

## 7. Metrik

- **Baris dihapus per episode** (utama, paling stabil).
- **Skor game** (dengan bonus Tetris).
- **Jumlah piece bertahan** (dengan batas 10.000).
- **Proporsi Tetris** (4 baris sekaligus) dari total pembersihan.
- **Kualitas tumpukan:** rata-rata lubang dan tinggi maksimum.
- **Sample efficiency:** langkah interaksi untuk mencapai ambang performa tertentu.
- Dilaporkan sebagai IQM dengan interval kepercayaan bootstrap antar seed (minimal 5 seed), plus proporsi run yang menyentuh batas episode dan kurva survival (lihat 3.5).

---

## 8. Interpretabilitas (arah C)

Dilakukan setelah agen terlatih, tanpa eksperimen pelatihan baru:
- **Probing value function (prioritas utama):** regresi linear dari aktivasi hidden layer ke fitur seperti jumlah lubang, tinggi kolom, bumpiness, dan sumur kosong. Menunjukkan konsep apa yang dikodekan jaringan.
- **Intervensi (prioritas utama):** ubah papan secara terkontrol (tambah lubang, naikkan kolom) dan lihat apakah nilai serta keputusan berubah sesuai dugaan dari hasil probing.
- **Saliency/Grad-CAM (tambahan, bobot kecil):** pada grid biner 20x10 hasilnya kurang informatif dibanding probing, jadi hanya pelengkap.
- **Analisis perilaku:** apakah agen menyisakan "sumur" untuk Tetris, bagaimana ia menangani piece S/Z, dan kapan ia gagal.
- **Intervensi:** ubah papan secara terkontrol dan lihat perubahan keputusan.

---

## 9. Visualisasi dan Demo (arah E)

Prinsip: **environment dan tampilan dipisah**. Pelatihan berjalan tanpa tampilan. Jangan menggambar di dalam loop pelatihan.

| Cara | Bentuk | Kegunaan |
|---|---|---|
| Viewer langsung | Jendela pygame di proses terpisah yang memuat checkpoint terbaru tiap beberapa detik | Menonton agen membaik hampir real-time |
| Rekaman per checkpoint | GIF/MP4 agen pada langkah 0, 10 rb, 100 rb, 1 jt, dst. | Video "evolusi belajar" untuk portofolio |
| Kurva metrik | TensorBoard | Bukti kuantitatif |
| Overlay analisis | Saliency/nilai di atas papan | Bagian interpretabilitas |
| Mode manusia vs agen | Game yang bisa dimainkan | Demo interaktif |

Renderer menganimasikan piece jatuh ke posisi tujuan sebagai efek visual saja. Logika tetap satu langkah per piece.

---

## 10. Struktur Repo

```
tetris-rl/
├── README.md                    # ringkasan, GIF demo, tabel hasil utama, cara setup dan reproduksi
├── LICENSE
├── pyproject.toml               # dependensi dengan versi dipatok
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml            # opsional: jalankan pytest otomatis
│
├── tetris_rl/                   # paket utama
│   ├── __init__.py
│   ├── pieces.py                # 7 piece dan rotasi unik
│   ├── board.py                 # versi referensi NumPy: tumbukan, hard drop, hapus baris
│   ├── board_fast.py            # versi numba (dites setara dengan board.py)
│   ├── env.py                   # TetrisEnv (Gymnasium), action mask, parameter varian
│   ├── features.py              # fitur buatan tangan (lubang, tinggi, bumpiness, dst.)
│   ├── render_text.py           # tampilan teks di terminal
│   ├── render_pygame.py         # tampilan grafis (animasi jatuh)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── random_agent.py
│   │   ├── heuristic.py         # heuristik berbobot
│   │   ├── cem.py               # CEM (fitur linear dan jaringan)
│   │   ├── td_afterstate.py     # TD/DQN afterstate (fitur linear dan jaringan)
│   │   └── ppo_masked.py        # PPO dengan action masking
│   ├── eval/
│   │   ├── evaluate.py          # protokol evaluasi (seed terpisah, batas 10.000 piece)
│   │   └── stats.py             # IQM, bootstrap CI (rliable), kurva survival
│   └── analysis/
│       ├── probing.py           # regresi linear hidden layer ke fitur
│       ├── intervention.py      # modifikasi papan terkontrol
│       └── saliency.py          # tambahan, bobot kecil
│
├── tests/
│   ├── test_pieces.py           # jumlah rotasi unik, 4 sel per rotasi
│   ├── test_board.py            # hapus 1-4 baris, game over, tumbukan
│   ├── test_board_equiv.py      # versi numba setara dengan referensi pada papan acak
│   ├── test_env.py              # check_env, determinisme seed, action mask
│   └── test_stats.py
│
├── configs/                     # satu file per eksperimen, lengkap dengan seed
│   ├── baseline_random.yaml
│   ├── baseline_heuristic.yaml
│   ├── e1_cem_features.yaml
│   ├── e2_cem_net.yaml
│   ├── e2_td_features.yaml
│   ├── e2_td_net.yaml
│   ├── e3_ppo_masked.yaml
│   └── pilot_board_height.yaml
│
├── scripts/
│   ├── train.py                 # train.py --config configs/xxx.yaml
│   ├── evaluate.py
│   ├── aggregate.py             # raw -> results/summary (CSV, tabel, grafik)
│   ├── record.py                # rekam GIF/video per checkpoint
│   └── view_live.py             # stretch goal: viewer langsung
│
├── results/
│   ├── raw/                     # DIABAIKAN git: log mentah, checkpoint tengah
│   └── summary/                 # DI-COMMIT: CSV agregat, tabel akhir, grafik
│
├── models/
│   └── final/                   # DI-COMMIT: checkpoint akhir yang kecil
│
├── docs/
│   └── assets/                  # DI-COMMIT: GIF demo untuk README
│
└── report/
    ├── decisions.md             # catatan keputusan dan penyimpangan, dengan tanggal
    └── report.md                # laporan akhir (atau PDF)
```

---

## 11. Setup di Ubuntu

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install numpy gymnasium torch stable-baselines3 sb3-contrib \
            tensorboard pygame imageio matplotlib pytest numba rliable
```

Catatan: PyTorch versi CPU sudah cukup untuk proyek ini. Pasang sesuai petunjuk resmi PyTorch jika ingin versi tertentu.

---

## 12. Roadmap (satu komponen per tahap)

| Tahap | Isi | Selesai jika |
|---|---|---|
| 1 | Kerangka papan dan definisi piece | 7 piece dengan jumlah rotasi benar (1/2/4), 4 sel per rotasi |
| 2 | Logika penempatan: hard drop, validasi, hapus baris, game over. Versi referensi NumPy murni lebih dulu, lalu versi **numba** untuk fungsi panas (tumbukan, hapus baris, `valid_placements`) | Lolos pengecekan manual di terminal, dan versi numba setara dengan versi referensi pada ribuan papan acak |
| 3 | Tes otomatis | Rotasi, multi-baris, game over, determinisme seed lolos |
| 4 | Renderer (teks lalu pygame) | Papan terlihat benar, animasi jatuh berjalan |
| 5 | Antarmuka Gymnasium | `check_env` lolos, action mask benar |
| 6 | Baseline (acak, heuristik, CEM) dengan visual | Angka E1 tersedia, video "sebelum" |
| 7 | Agen deep RL (DQN afterstate, PPO masking) | Kurva belajar dan checkpoint tersimpan |
| 8 | Eksperimen E2 sampai E5 | Tabel hasil multi-seed |
| 9 | Interpretabilitas | Visual dan analisis perilaku |
| 10 | Demo dan laporan | README, video, grafik, analisis kegagalan |

---

## 13. Risiko dan Mitigasi

| Risiko | Mitigasi |
|---|---|
| Bug aturan memengaruhi seluruh hasil | Tes otomatis sejak tahap 3, bandingkan dengan satu implementasi referensi hanya untuk memeriksa aturan |
| Deep RL tidak belajar (reward jarang) | Mulai dari afterstate dan fitur, tambahkan shaping bertahap, cek dulu dengan papan kecil |
| Hasil bervariasi antar seed | Minimal 5 seed, banyak episode evaluasi, laporkan IQM dan CI bootstrap |
| Ceiling effect (semua tamat di batas 10.000) | Pilot tinggi papan, data tersensor dan kurva survival (3.5) |
| Environment Python lambat (bottleneck CEM) | Numba untuk fungsi panas sejak awal, tes kesetaraan dengan versi referensi |
| Evaluasi tidak berakhir | Batas 10.000 piece |
| Perbandingan tidak adil | Anggaran langkah interaksi dan tuning disamakan, dicatat |
| Terlihat seperti tutorial | Pertanyaan riset jelas, ablation, analisis kegagalan, interpretabilitas |
| Cakupan melebar | Arah D dan E6 hanya jika inti selesai |

---

## 13.1 Prioritas dan Stretch Goal

**Inti (wajib):** E1 sampai E5 (termasuk faktorial 2x2), probing + intervensi, rekaman GIF per checkpoint, laporan.

**Stretch goal (hanya jika inti selesai):** E6 (per tombol vs penempatan akhir), E7 (generalisasi varian aturan), mode manusia vs agen, dan viewer pygame langsung. Untuk demo portofolio, rekaman GIF per checkpoint sudah cukup.

---

## 14. Deliverable Akhir

- Repo GitHub dengan kode, tes, dan konfigurasi eksperimen yang reproduktibel.
- Laporan singkat: pertanyaan, metode, hasil dengan tabel dan grafik multi-seed, analisis kegagalan, keterbatasan (simplifikasi di bagian 3.2).
- Video evolusi belajar agen dan perbandingan dengan baseline.
- Demo manusia vs agen.

---

## 15. Referensi Awal

- Szita & Lorincz (2006), *Learning Tetris Using the Noisy Cross-Entropy Method*.
- Gabillon, Ghavamzadeh & Scherrer (2013), *Approximate Dynamic Programming Finally Performs Well in the Game of Tetris*.
- Fitur Dellacherie untuk Tetris (tinggi kolom, lubang, kebergelombangan, dan sejenisnya).
- Dokumentasi Gymnasium, Stable-Baselines3, sb3-contrib (`MaskablePPO`), CleanRL.

Periksa sendiri detail dan angka pada sumber aslinya sebelum dikutip di laporan.
