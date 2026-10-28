# Spesifikasi Teknis Lingkungan Tetris RL (`TetrisEnv`)

Dokumen ini mendefinisikan secara menyeluruh spesifikasi antarmuka lingkungan `TetrisEnv` berbasis standar Gymnasium. Dokumen ini dibuat sesuai arahan dokumen standar proyek (`Agent/DOCUMENTATION_STANDARDS.md`).

---

## 1. Aturan Game & Transisi
- **Grid Papan:** Ukuran standar $20 \times 10$ ($H=20, W=10$) bertipe `np.uint8` (0 = kosong, 1 = terisi).
- **Mekanisme Gerak (Final Placement):**
  Agen memilih kombinasi $(\text{rotasi}, \text{kolom})$ sebagai satu aksi terpadu. Piece dimunculkan dari atas dan langsung jatuh bebas (*hard drop*) hingga menumpu pada dasar papan atau puncak tumpukan blok di bawahnya.
- **Pembersihan Baris (Line Clears):**
  Setiap baris yang 10 selnya terisi penuh ($100\%$) dihapus. Baris-baris di atasnya turun secara gravitasi lurus ke bawah. Sebanyak 1, 2, 3, atau 4 baris (*Tetris*) dapat terhapus secara simultan.
- **Game Over:**
  Terjadi ketika piece yang dijatuhkan tidak muat di dalam batas atas papan (artinya posisi pendaratan pojok kiri atas piece $r_{\text{landing}} < 0$).

---

## 2. Ruang Aksi & Pendekodean (`action_space`)
- **Tipe Ruang Aksi:** `gymnasium.spaces.Discrete(40)` (untuk lebar papan 10).
- **Rumus Pendekodean:**
  $$\text{rotasi} = \lfloor a / 10 \rfloor$$
  $$\text{kolom} = a \pmod{10}$$
- **Rentang:**
  - $\text{rotasi} \in [0, 3]$
  - $\text{kolom} \in [0, 9]$

---

## 3. Action Masking (`info["action_mask"]` / `env.action_masks()`)
Untuk setiap piece aktif, vektor boolean 1D berukuran 40 dihasilkan untuk memfilter aksi yang valid:
1. **Pencegahan Keluar Papan (Out-of-Bounds):**
   Aksi dengan $\text{kolom} > W - \text{width}_{\text{piece}}(\text{rotasi})$ di-mask bernilai `False`.
2. **Eliminasi Rotasi Duplikat (Simetri Geometri):**
   - Tetromino `O`: hanya 1 rotasi unik (indeks 0). Rotasi 1, 2, 3 di-mask `False`.
   - Tetromino `I`, `S`, `Z`: hanya 2 rotasi unik (indeks 0 dan 1). Rotasi 2, 3 di-mask `False`.
   - Tetromino `T`, `J`, `L`: 4 rotasi unik (indeks 0, 1, 2, 3).
3. **Penalti Aksi Ilegal:**
   Jika agen melanggar mask dan mengeksekusi aksi ilegal, environment seketika menghentikan episode (`terminated=True`) dan memberikan penalti sebesar $-50.0$.

---

## 4. Ruang Observasi (`observation_space`)
Tersedia tiga varian representasi keadaan:

| Opsi (`obs_type`) | Bentuk Tensor | Isi & Format |
|---|---|---|
| `'raw'` | `Box(shape=(214,), dtype=float32)` | Papan biner diratakan ($20 \times 10 = 200$) + One-hot Piece Aktif (7) + One-hot Next Piece (7). Rentang $[0.0, 1.0]$. |
| `'features'` | `Box(shape=(29,), dtype=float32)` | Fitur Buatan Tangan ($10 + 5 = 15$) + One-hot Piece Aktif (7) + One-hot Next Piece (7). |
| `'combined'` | `Box(shape=(229,), dtype=float32)` | Papan biner (200) + Fitur Buatan Tangan (15) + One-hot Aktif (7) + One-hot Next (7). |

### Rincian 15 Fitur Buatan Tangan:
1. `col_0` s.d. `col_9`: Ketinggian tumpukan di setiap kolom ($0 \dots 20$).
2. `max_height`: Ketinggian maksimum tumpukan.
3. `aggregate_height`: Total jumlah ketinggian seluruh kolom.
4. `holes`: Jumlah sel kosong yang memiliki minimal satu blok terisi di atasnya pada kolom yang sama.
5. `bumpiness`: Total selisih mutlak ketinggian antar kolom bersebelahan $\sum_{c=0}^{W-2} |h_c - h_{c+1}|$.
6. `lines_cleared`: Jumlah baris yang terhapus pada langkah terkait.

---

## 5. Skema Reward Pelatihan (`reward_type`)

Poin resmi permainan (*Game Score*) dihitung tetap:
- 1 baris = 100 poin
- 2 baris = 300 poin
- 3 baris = 500 poin
- 4 baris = 800 poin (Tetris)

Varian reward pelatihan:
- **`R0` (Sparse Game Points):**
  $$R = \text{GamePoints}(\text{lines cleared})$$
- **`R1` (Survival Bonus):**
  $$R = \text{GamePoints} + 1.0 \cdot \mathbb{I}(\text{not game over}) - 10.0 \cdot \mathbb{I}(\text{game over})$$
- **`R2` (Heuristic Penalties):**
  $$R = R_1 - 0.5 \cdot \text{holes} - 0.1 \cdot \text{max height} - 20.0 \cdot \mathbb{I}(\text{game over})$$
- **`R3` (Potential-Based Reward Shaping):**
  Menggunakan fungsi potensial $\Phi(s) = - (4.0 \cdot \text{holes} + 1.0 \cdot \text{bumpiness} + 0.5 \cdot \text{aggregate height})$:
  $$R = R_0 + (\Phi(s') - \Phi(s)) - 20.0 \cdot \mathbb{I}(\text{game over})$$
  *(Berdasarkan teorema Ng, Harada & Russell 1999, potential-based shaping menjamin policy optimal tetap invarian).*

> [!IMPORTANT]
> Evaluasi akhir seluruh agen **wajib menggunakan skor game resmi** (`info["score"]`), bukan akumulasi reward pelatihan.

---

## 6. Kondisi Terminasi & Pemotongan
- **`terminated = True`:** Terjadi saat $r_{\text{landing}} < 0$ (tumpukan melebihi batas atas papan) atau saat aksi ilegal dieksekusi.
- **`truncated = True`:** Terjadi saat total piece yang dimainkan (`steps`) mencapai batas maksimal `max_steps` (default 10.000 untuk pengujian data tersensor).

---

## 7. Isi Dictionary `info`
Setiap panggilan `reset()` dan `step()` mengembalikan `info` dengan metadata lengkap:
- `action_mask`: `np.ndarray` boolean (40,)
- `score`: `int` (akumulasi skor game resmi)
- `lines_cleared`: `int` (total baris dihapus sejauh ini)
- `step_lines_cleared`: `int` (baris dihapus pada langkah ini)
- `holes`: `int` (jumlah lubang saat ini)
- `max_height`: `int` (tinggi puncak tumpukan)
- `bumpiness`: `int` (tingkat kebergelombangan)
- `aggregate_height`: `int` (total tinggi kolom)
- `steps`: `int` (jumlah langkah/piece)
- `current_piece`: `int` (ID piece aktif 0..6)
- `next_piece`: `int` (ID piece berikutnya 0..6)
- `is_game_over`: `bool`

---

## 8. Daftar Simplifikasi Aturan Game
Sesuai Bagian 3.2 Dokumen Proyek, hal-hal berikut **secara eksplisit tidak dimodelkan**:
- Gerakan *tuck* (sliding piece ke bawah rongga).
- *T-spin* dan *mini T-spin*.
- *Wall kick* (SRS offset tables).
- Kecepatan jatuh / gravitasi bertahap (frame-by-frame drop).
- Fitur *hold*.
- *Garbage lines* dari luar.

