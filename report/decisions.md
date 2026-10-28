# Log Keputusan Teknis dan Perubahan Desain

Dokumen ini mencatat keputusan arsitektural, teknis, dan penyimpangan dari *Dokumen Proyek* beserta tanggal dan alasannya, sesuai aturan Bagian 11 & 12 dari `AGENT_CONTRACT.md`.

---

## 2026-10-28: Inisialisasi Proyek dan Setup Dasar
- **Keputusan:**
  - Menetapkan Python 3.12 (lingkungan lokal sistem) menggunakan `.venv` terisolasi.
  - Memisahkan definisi piece dalam representasi matriks biner 2D rapat (*tight bounding box*) dan koordinat relatif `(4, 2)`.
  - Urutan standar tetromino ditetapkan: `I` (0), `O` (1), `T` (2), `S` (3), `Z` (4), `J` (5), `L` (6).
  - Rotasi unik dikunci sesuai Dokumen Proyek Bagian 3.1:
    - `O`: 1 rotasi
    - `I`, `S`, `Z`: 2 rotasi unik
    - `T`, `J`, `L`: 4 rotasi unik
  - Semua sel per rotasi bernilai tepat 4 blok (`np.uint8`).
  - Menambahkan isolasi plugin (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`) di `.venv/bin/activate` untuk mencegah tabrakan plugin ROS 2 Jazzy yang ada di host OS.

