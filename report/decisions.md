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