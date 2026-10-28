# Tetris RL: Algoritma atau Representasi? CEM vs Deep RL di Tetris

Proyek riset portofolio Reinforcement Learning untuk menganalisis apakah selisih performa antara metode klasik berbasis fitur buatan tangan (handcrafted features) dan Deep RL pada game Tetris berasal dari **algoritma** (gradient-free vs TD/gradient) atau dari **representasi** (fitur linear vs papan mentah).

---

## 1. Setup Lingkungan (Ubuntu / Linux)

### Kebutuhan Sistem:
- Python 3.10+ (teruji pada Python 3.12.3)
- Tanpa kebutuhan GPU (cukup CPU)

### Langkah Setup:
```bash
# 1. Buat virtual environment
python3 -m venv .venv

# 2. Aktifkan virtual environment
source .venv/bin/activate

# 3. Install paket dan dependensi pengembangan
pip install -e ".[dev]"
```

*Catatan:* Jika sistem host memiliki ROS 2 terpasang, environment virtual telah dikonfigurasi untuk mengabaikan plugin eksternal melalui variabel `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

---

## 2. Menjalankan Pengujian (Testing)

Untuk memvalidasi seluruh logika game, aturan, dan komponen:

```bash
source .venv/bin/activate
pytest -v
```

---

## 3. Catatan Penyederhanaan Aturan Game

Sesuai metodologi riset Tetris AI standar dan Dokumen Proyek (Bagian 3.2):
- **Ruang Aksi:** Penempatan akhir (rotasi + kolom), hard drop langsung.
- **Tidak dimodelkan:** Gerakan *tuck* (geser di bawah overhang), *T-spin*, *wall kick*, kecepatan jatuh/gravitasi bertahap, dan *hold*.
