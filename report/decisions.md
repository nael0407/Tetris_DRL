# Log Keputusan Teknis dan Perubahan Desain

Dokumen ini mencatat keputusan arsitektural, teknis, dan penyimpangan dari *Dokumen Proyek* beserta tanggal dan alasannya, sesuai aturan Bagian 11 & 12 dari `AGENT_CONTRACT.md` dan `Agent/DOCUMENTATION_STANDARDS.md`.

Format per entri:
- **Tanggal:** YYYY-MM-DD
- **Keputusan:** Keputusan teknis/arsitektur yang diambil.
- **Alasan:** Justifikasi mengapa keputusan tersebut dipilih.
- **Alternatif yang ditolak:** Pendekatan lain yang dipertimbangkan beserta alasan penolakannya.
- **Dampak:** Konsekuensi terhadap kode, eksperimen, atau alur kerja.

---

## 2026-10-28: Inisialisasi Environment & Isolasi Pytest dari ROS 2
- **Tanggal:** 2026-10-28
- **Keputusan:** Menggunakan `.venv` berbasis Python 3.12 lokal dan menambahkan `export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` pada skrip aktivasi environment (`.venv/bin/activate`).
- **Alasan:** Host OS Ubuntu memiliki instalasi ROS 2 Jazzy yang mengekspor path ke `PYTHONPATH`, menyebabkan plugin `launch_testing_ros` dipanggil secara otomatis oleh pytest dan menimbulkan `PluginValidationError`.
- **Alternatif yang ditolak:** Menghapus atau memodifikasi instalasi ROS 2 di tingkat sistem (melanggar prinsip larangan tindakan destruktif pada sistem host).
- **Dampak:** Pengujian unit `pytest` dapat dijalankan dengan bersih, cepat, dan tanpa konflik plugin eksternal sistem.

---

## 2026-10-28: Representasi Tetromino & Eliminasi Rotasi Simetris (Tahap 1)
- **Tanggal:** 2026-10-28
- **Keputusan:** Mengunci jumlah rotasi hanya pada rotasi unik (O: 1, I/S/Z: 2, T/J/L: 4; total 19 konfigurasi) menggunakan *dual representation* (matriks biner 2D rapat *tight bounding box* dan koordinat offset relatif `(4, 2)` int32), serta menyiapkan lookup table statis `COORDS_TABLE` dan `DIMS_TABLE`.
- **Alasan:** Mencegah redundansi pada *action space* (Discrete(40)) dan action masking, serta memastikan fungsi penempatan cepat (Tahap 2 dengan Numba) dapat mengakses dimensi dan koordinat tanpa overhead objek Python dinamis.
- **Alternatif yang ditolak:**
  - Mengizinkan 4 rotasi untuk semua piece termasuk O/I/S/Z (ditolak karena membuat policy melihat aksi berbeda nomor yang perilakunya 100% identik).
  - Menyimpan bentuk dalam grid 4x4 dengan padding kosong (ditolak karena memperumit kalkulasi kolom legal dan batas dinding).
- **Dampak:** Pengecekan kolom legal menjadi deterministik ($0 \le \text{kolom} \le 10 - \text{width}$), dan action masking di tahap selanjutnya dapat memotong aksi di atas rotasi unik secara tepat.

---

## 2026-10-28: Adopsi Standar Dokumentasi Proyek
- **Tanggal:** 2026-10-28
- **Keputusan:** Mengadopsi panduan dokumen wajib (`README.md`, `report/report.md`, `report/decisions.md`, docstring) dan sangat disarankan (`docs/environment.md`, `docs/experiments.md`, komentar YAML) yang dicatat di `Agent/DOCUMENTATION_STANDARDS.md`.
- **Alasan:** Menjaga transparansi, keteraturan, dan kualitas portofolio riset dari awal hingga akhir proyek.
- **Alternatif yang ditolak:** Menulis dokumentasi dan laporan hanya di akhir setelah seluruh eksperimen selesai (ditolak karena rawan lupa detail desain, asumsi awal, atau bug yang terjadi selama proses).
- **Dampak:** Kerangka `report/report.md` akan mulai disusun lebih awal (khususnya bagian Pendahuluan dan Metode), dan setiap file konfigurasi eksperimen akan memiliki metadata yang konsisten.

---

## 2026-10-28: Arsitektur Logika Papan & Akselerasi Numba (Tahap 2)
- **Tanggal:** 2026-10-28
- **Keputusan:** Membangun dua implementasi logika papan: `tetris_rl/board.py` (referensi NumPy murni) dan `tetris_rl/board_fast.py` (terakselerasi Numba JIT), dengan perhitungan pendaratan langsung dari profil ketinggian kolom (`r_landing = min(top[col + dc] - 1 - dr)`) dan pembersihan baris menggunakan algoritma *two-pointer compaction* dari bawah ke atas.
- **Alasan:** Pelatihan agen berbasis afterstate (DQN dan CEM) memerlukan evaluasi ribuan papan per detik. Pendekatan analitis profil kolom menghilangkan kebutuhan simulasi loop pergeseran sel bertahap yang lambat, sementara Numba mengompilasi eksekusi afterstate menjadi instruksi mesin langsung.
- **Alternatif yang ditolak:**
  - Melakukan simulasi jatuh per langkah (step-by-step falling loop) untuk hard drop (ditolak karena tidak diperlukan pada model penempatan akhir tanpa tuck dan membebani komputasi).
  - Hanya menulis versi Numba tanpa versi referensi NumPy murni (ditolak karena melanggar prinsip verifikasi kontrak kerja: kebenaran logika Numba wajib dibuktikan setara dengan kode referensi murni).
- **Dampak:** Perilaku NumPy dan Numba terbukti 100% identik pada pengujian 2.500 variasi papan acak (`test_board_equiv.py`), sekaligus menyediakan fungsi batch afterstate berkecepatan tinggi untuk tahapan algoritma RL.

---

## 2026-10-28: Pengalihan Operasi Git ke Pemilik Proyek
- **Tanggal:** 2026-10-28
- **Keputusan:** Operasi `git add`, `git commit`, dan `git push` sepenuhnya dieksekusi secara manual oleh pemilik proyek (Natanael). AI Agent hanya bertugas menyiapkan dan menyajikan perintah Git yang tepat, terstruktur, dan siap salin-tempel di setiap akhir laporan tahap.
- **Alasan:** Instruksi langsung dari pemilik proyek untuk memegang kendali penuh atas manajemen versi lokal dan repositori GitHub.
- **Alternatif yang ditolak:** Agen mengeksekusi git commit/push otomatis via bash tool.
- **Dampak:** Agen tidak akan lagi menjalankan perintah `git add`, `git commit`, atau `git push` secara mandiri.

---

## 2026-10-28: Pengujian Otomatis Lanjutan & Penataan Engine (Tahap 3)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Menambahkan modul generator piece berbasis seed (`UniformPieceGenerator` dan `SevenBagPieceGenerator`) di `tetris_rl/pieces.py`.
  2. Menyertakan field `landing_row: int` pada `ValidPlacement` di `board.py` dan `board_fast.py`.
  3. Memperbaiki cakupan pemindaian line-clear pada `board.py`: mengubah pemindaian lokal rentang piece menjadi pemindaian penuh papan (`range(height)`) agar 100% konsisten dengan kernel JIT `board_fast.py`.
- **Alasan:**
  1. Reproduktibilitas riset menuntut urutan kemunculan piece dikontrol ketat oleh random seed generator terisolasi.
  2. Nilai `landing_row` sangat dibutuhkan oleh heuristik (fitur landing height Dellacherie) dan pemilihan aksi deterministik.
  3. Menjamin kesetaraan absolut antara versi NumPy dan Numba bahkan pada kasus papan uji sintetis yang memiliki baris penuh di luar area pendaratan piece.
- **Alternatif yang ditolak:**
  - Menggunakan modul `random` bawaan Python untuk generator piece (ditolak karena tidak thread-safe dan sulit diintegrasikan dengan kontrol seed NumPy).
  - Membiarkan `ValidPlacement` tanpa `landing_row` (ditolak karena memaksa agen menghitung ulang pendaratan yang sebenarnya sudah didapat saat penempatan).
- **Dampak:** Seluruh 33 unit test (mencakup rotasi, multi-baris tak bersambung, presisi batas game over, preservasi lubang, dan simulasi deterministik multi-seed) lulus 100%.

---

## 2026-10-28: Pengujian Otomatis Lanjutan & Penataan Engine (Tahap 3)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Menambahkan modul generator piece berbasis seed (`UniformPieceGenerator` dan `SevenBagPieceGenerator`) di `tetris_rl/pieces.py`.
  2. Menyertakan field `landing_row: int` pada `ValidPlacement` di `board.py` dan `board_fast.py`.
  3. Memperbaiki cakupan pemindaian line-clear pada `board.py`: mengubah pemindaian lokal rentang piece menjadi pemindaian penuh papan (`range(height)`) agar 100% konsisten dengan kernel JIT `board_fast.py`.
- **Alasan:**
  1. Reproduktibilitas riset menuntut urutan kemunculan piece dikontrol ketat oleh random seed generator terisolasi.
  2. Nilai `landing_row` sangat dibutuhkan oleh heuristik (fitur landing height Dellacherie) dan pemilihan aksi deterministik.
  3. Menjamin kesetaraan absolut antara versi NumPy dan Numba bahkan pada kasus papan uji sintetis yang memiliki baris penuh di luar area pendaratan piece.
- **Alternatif yang ditolak:**
  - Menggunakan modul `random` bawaan Python untuk generator piece (ditolak karena tidak thread-safe dan sulit diintegrasikan dengan kontrol seed NumPy).
  - Membiarkan `ValidPlacement` tanpa `landing_row` (ditolak karena memaksa agen menghitung ulang pendaratan yang sebenarnya sudah didapat saat penempatan).
- **Dampak:** Seluruh 33 unit test (mencakup rotasi, multi-baris tak bersambung, presisi batas game over, preservasi lubang, dan simulasi deterministik multi-seed) lulus 100%.

---

## 2026-10-28: Arsitektur Renderer Teks & Pygame (Tahap 4)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Memisahkan visualisasi menjadi dua modul independen: `tetris_rl/render_text.py` (terminal ASCII/Unicode) dan `tetris_rl/render_pygame.py` (grafis Pygame 2D).
  2. Mengimplementasikan mode headless otomatis (`SDL_VIDEODRIVER=dummy`) pada `PygameRenderer` jika `DISPLAY` tidak terpasang atau saat flag `headless=True` aktif.
  3. Mengimplementasikan fungsi `render_drop_animation` yang menghasilkan deretan frame RGB meluncur dari atas ke posisi landing row sebelum mengeksekusi frame papan hasil, serta fungsi `save_gif` berbasis `imageio`.
- **Alasan:**
  1. Menjaga komitmen Bagian 9 Dokumen Proyek: visualisasi tidak boleh memperlambat atau berada di dalam loop pelatihan agen.
  2. Memungkinkan pengujian unit otomatis di CI/lingkungan server Linux headless tanpa X11 display server.
  3. Memenuhi kebutuhan portofolio untuk merekam GIF/video evolusi belajar agen per checkpoint.
- **Alternatif yang ditolak:**
  - Menggabungkan kode pygame langsung di dalam `TetrisEnv.step()` (ditolak karena membuat pelatihan menjadi ribuan kali lebih lambat dan merusak arsitektur modular).
  - Menggunakan library video encoding berat seperti OpenCV (ditolak karena `imageio` + `pillow` jauh lebih ringan dan sesuai dengan stack Python murni proyek).
- **Dampak:** Tersedia visualisasi teks cepat di terminal dan kemampuan merekam rekaman visual animasi piece ke format GIF/MP4 tanpa monitor fisik. File demo visual berhasil di-generate di `docs/assets/demo_stage4.gif`.

---

## 2026-10-28: Arsitektur Antarmuka Gymnasium TetrisEnv (Tahap 5)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Mengimplementasikan `TetrisEnv` mematuhi standar resmi Gymnasium dengan ruang aksi `Discrete(40)` dan observasi 1D `Box(float32)` untuk ketiga mode ('raw': 214 dimensi, 'features': 29 dimensi, 'combined': 229 dimensi), masing-masing memuat one-hot encoding piece aktif (7) dan preview next piece (7).
  2. Mengimplementasikan 4 varian reward pelatihan (R0 sampai R3) dengan reward shaping R3 menggunakan formulasi *potential-based* ($\Phi(s) = - (4 \cdot holes + 1 \cdot bumpiness + 0.5 \cdot aggregate\_height)$) yang secara teoretis menjamin invariansi kebijakan optimal.
  3. Mengunci metrik evaluasi pada skor game asli (100/300/500/800) yang selalu dicatat pada `info["score"]`.
  4. Menyediakan metode helper `action_masks()` untuk integrasi langsung tanpa wrapper tambahan dengan `sb3-contrib.MaskablePPO`.
  5. Menulis dokumen spesifikasi lengkap di `docs/environment.md`.
- **Alasan:**
  1. Vektor 1D Box kompatibel *out-of-the-box* dengan seluruh pustaka RL standar (Stable-Baselines3, CleanRL, PyTorch) tanpa memerlukan arsitektur *custom feature extractor* yang rumit.
  2. Memenuhi persyaratan eksperimen Dokumen Proyek Bagian 4.3, 4.4, dan 5.4.
  3. Memastikan pemenuhan standar dokumen wajib dan terverifikasi 100% lolos `gymnasium.utils.env_checker.check_env`.
- **Alternatif yang ditolak:**
  - Menggunakan `spaces.Dict` sebagai observasi default (ditolak untuk eksperimen utama karena memerlukan `MultiInputPolicy` yang lebih lambat dan memiliki overhead pemrosesan kamus per langkah pada MLP).
  - Memberi reward negatif tanpa terminasi pada aksi ilegal (ditolak karena membiarkan agen berputar-putar mencoba aksi ilegal yang sia-sia).
- **Dampak:** Seluruh 48 unit test (termasuk `check_env` pada 3 mode observasi) lulus 100%. Lingkungan siap digunakan untuk melatih dan mengevaluasi agen baseline di Tahap 6.

---

## 2026-10-28: Evaluasi Baseline Klasik & Validasi Ceiling Effect (Tahap 6)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Mengimplementasikan 3 baseline klasik: `RandomAgent` (batas bawah), `HeuristicAgent` (bobot Dellacherie), dan `CEMAgent` (Cross-Entropy Method dengan pelacakan piece interaksi riil).
  2. Menstandarisasi protokol evaluasi multi-seed (5 seed evaluasi independen x 5 episode = 25 episode) dengan metrik IQM dan bootstrap CI 95% (`tetris_rl/eval/stats.py`).
  3. Mengonfirmasi hasil empiris batas papan: agen heuristik terbaik bertahan rata-rata 1.440 langkah (maksimal di kisaran ~2.500 langkah) dengan rasio tersensor $0.0\%$ terhadap batas 10.000 piece. Dengan demikian, papan standar $10 \times 20$ diputuskan tetap sebagai ukuran utama eksperimen karena terbukti bebas dari risiko *ceiling effect*.
- **Alasan:**
  1. Memberikan baseline komparatif yang kredibel untuk menjawab pertanyaan riset "algoritma atau representasi" pada desain faktorial $2 \times 2$.
  2. Memenuhi aturan kejujuran kontrak kerja: total piece CEM dihitung dari seluruh evaluasi kandidat ($659.779$ piece), bukan sekadar nomor generasi.
  3. Menjawab pertanyaan pilot Bagian 3.5 Dokumen Proyek secara berbasis data nyata (*data-driven*).
- **Alternatif yang ditolak:**
  - Memperkecil tinggi papan menjadi 10x12 (ditolak karena hasil empiris membuktikan pada 10x20 agen heuristik belum mengalami ceiling effect).
  - Mengevaluasi hanya dengan 1 seed atau melaporkan mean +/- std deviasi saja (ditolak karena rentan terhadap skewness dan outlier).
- **Dampak:** Data baseline E1 lengkap tersimpan di `results/summary/` (JSON), rekaman GIF visual "sebelum belajar" dan "sesudah" tersedia di `docs/assets/`, dan katalog eksperimen tercatat di `docs/experiments.md`.

---

## 2026-10-28: Perbaikan Skrip Perekaman Visualisasi Gameplay (record.py)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Mengubah argumen `--output` menjadi opsional dengan default `"docs/assets/demo.gif"`.
  2. Menambahkan auto-discovery bobot CEM: jika `--agent cem` dipilih tanpa argumen `--weights`, skrip otomatis mencari dan memuat `models/final/cem_features_best.npy`.
  3. Menyesuaikan dimensi default papan ke ukuran standar Tetris $10 \times 20$ (sebelumnya `16`), dan menyediakan flag konfigurasi `--height` serta `--width`.
  4. Memperbaiki anotasi tipe `weights_path: Optional[str] = None` dan proteksi pendaratan `effective_landing_row = max(0, res_preview.landing_row)` untuk mencegah crash saat move menyebabkan game over.
- **Alasan:** Menghilangkan error CLI saat skrip dipanggil tanpa flag `--output`, mencegah konflik type checker pada VS Code/Pylance, dan memastikan rekaman sesuai standar dimensi game 10x20.
- **Alternatif yang ditolak:** Memaksa pengguna selalu mengetikkan path output yang panjang secara manual di terminal.
- **Dampak:** Skrip `record.py` dapat dieksekusi langsung tanpa argumen, kompatibel dengan seluruh agen (Random, Heuristic, CEM), dan terverifikasi pada suite pengujian `test_record_agent_gameplay_script`.

---

## 2026-10-28: Arsitektur Agen Deep RL & Verifikasi Smoke-Test (Tahap 7)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Menginstal PyTorch versi CPU (`torch-cpu`, ~196 MB) untuk menghemat kapasitas disk sistem dan menghilangkan overhead transfer CPU-GPU pada forward pass jaringan kecil berkecepatan tinggi.
  2. Mengimplementasikan `LinearTDAfterstateAgent` dengan pembaruan gradien ternormalisasi (*Normalized LMS / Normalized TD*) dan pembatasan klip langkah $\Delta \mathbf{w} \in [-1.0, 1.0]$ untuk menjamin stabilitas numerik pada fitur dengan skala dinamis.
  3. Mengimplementasikan `DQNAfterstateAgent` dengan arsitektur Double DQN, Experience Replay Buffer NumPy berkecepatan tinggi, target network update periodik, serta evaluasi batch cepat terhadap seluruh penempatan sah (`valid_placements`).
  4. Mengintegrasikan `MaskedPPOAgent` berbasis `sb3-contrib.MaskablePPO` dengan wrapper `ActionMasker` yang terhubung langsung ke `TetrisEnv.action_masks()`.
  5. Menambahkan logging otomatis ke TensorBoard, mekanisme checkpoint model (`models/checkpoints/` dan `models/final/`), serta memvalidasi pipeline pelatihan melalui *smoke-test* pilot run.
- **Alasan:**
  1. Memenuhi kriteria selesai Tahap 7 pada Dokumen Proyek: seluruh agen siap latih, kurva belajar tercatat di TensorBoard, dan checkpoint berhasil tersimpan.
  2. Mencegah *divergence* atau *overflow* pada linear TD(0) saat memproses fitur unnormalized.
  3. Memastikan kesiapan penuh seluruh sel eksperimen faktorial $2 \times 2$ (E2 s/d E5) sebelum eksekusi pelatihan skala besar di Tahap 8.
- **Alternatif yang ditolak:**
  - Memasang PyTorch CUDA dengan paket dependensi cuDNN/Triton 4GB+ (ditolak karena memicu `OSError: [Errno 28] No space left on device` pada partisi sistem host dan tidak memberikan akselerasi berarti untuk MLP berdimensi 200).
  - Melatih model langsung berjam-jam tanpa smoke-test (ditolak sesuai *AGENT_CONTRACT.md* untuk mencegah kegagalan runtime di tengah jalan).
- **Dampak:** 61 unit test lulus 100%, kurva belajar tercatat di `results/tensorboard/`, checkpoint dan model final tersimpan di `models/`, dan pipeline siap untuk Tahap 8.

---

## 2026-10-28: Eksekusi Eksperimen Faktorial 2x2 & Analisis Multi-Seed (Tahap 8)
- **Tanggal:** 2026-10-28
- **Keputusan:**
  1. Menjalankan seluruh matriks eksperimen faktorial $2 \times 2$ (E2): CEM vs TD pada Fitur Linear vs Papan Mentah 200 piksel biner (`e1_cem_features`, `e2_td_features`, `e2_cem_net`, `e2_td_net`).
  2. Mengeksekusi eksperimen perbandingan policy gradient ber-masking (E3: `e3_ppo_masked`) dan ablasi varian reward (E4: `e4_reward_r0`, `e4_reward_r1`, `e4_reward_r2`).
  3. Mengukur seluruh hasil akhir menggunakan protokol evaluasi multi-seed (5 seed evaluasi independen x 5 episode = 25 episode) dengan metrik Interquartile Mean (IQM) dan 95% Bootstrap Confidence Interval.
  4. Memperbarui tabel katalog hasil resmi pada `docs/experiments.md`.
- **Alasan:** Menjawab pertanyaan riset utama portofolio secara empiris dan membuktikan secara kuantitatif apakah perbedaan performa di Tetris berasal dari algoritma atau representasi.
- **Alternatif yang ditolak:** Hanya mengevaluasi 1 seed per metode atau hanya melaporkan skor rata-rata tanpa confidence interval (ditolak karena melanggar standar rigoritas riset RL).
- **Dampak:** Hipotesis terbukti secara kuat berbasis data empiris: kesenjangan performa didominasi oleh representasi (papan mentah menghasilkan skor 0 di kedua algoritma, sedangkan fitur linear menghasilkan skor tinggi). Tabel komparatif multi-seed lengkap tersimpan di `results/summary/` dan `docs/experiments.md`.


