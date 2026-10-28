# Standar dan Panduan Dokumentasi Proyek Tetris RL

Dokumen ini berisi pedoman wajib dan sangat disarankan terkait struktur, waktu pembuatan, serta isi setiap dokumen di dalam repositori Tetris RL.

---

## I. Dokumen Wajib

### 1. `README.md`
- **Waktu pengerjaan:** Kerangka disiapkan di awal; hasil akhir, tabel utama, dan GIF demo diisi setelah eksperimen selesai.
- **Isi wajib:**
  - Judul proyek dan satu kalimat pertanyaan riset inti.
  - GIF demo evolusi belajar agen (dari langkah awal hingga performa terbaik).
  - Hasil utama: satu tabel ringkasan, satu grafik komparasi, dan kesimpulan 1–2 kalimat yang jujur (termasuk jika hipotesis tidak terbukti).
  - Cakupan dan simplifikasi game (penempatan akhir, tanpa *tuck*, *T-spin*, *wall kick*, *hold*, gravitasi bertahap).
  - Petunjuk setup di Ubuntu (perintah *copy-paste* yang dapat langsung dieksekusi).
  - Cara menjalankan tes otomatis (`pytest`).
  - Cara mereproduksi satu eksperimen kecil (skala menit).
  - Struktur repositori singkat.
  - Lisensi dan daftar referensi pustaka/makalah.

### 2. `report/report.md` (Laporan Riset Utama)
- **Waktu pengerjaan:** Kerangka disiapkan sejak awal; bagian *Pendahuluan* & *Metode* ditulis sebelum eksperimen berjalan; bagian *Hasil* diisi setelah eksperimen tuntas.
- **Isi wajib:**
  - **Pendahuluan:** Motivasi riset dan pertanyaan riset utama.
  - **Latar Belakang Singkat:** Tetris sebagai benchmark RL, algoritma CEM (gradient-free), TD/DQN (gradient-based).
  - **Metode:** Spesifikasi environment, representasi observasi, ruang aksi & action masking, reward function, serta desain faktorial $2 \times 2$.
  - **Protokol Eksperimen:** Manajemen seed, anggaran langkah interaksi (definisi piece untuk CEM vs RL), prosedur evaluasi, dan pemilihan tinggi papan berdasarkan uji pilot.
  - **Hasil:** Eksperimen E1 sampai E5, tabel metrik IQM dengan bootstrap CI (`rliable`), proporsi run tersensor (mencapai batas 10.000 piece), dan kurva survival Kaplan-Meier.
  - **Interpretabilitas:** Probing representasi aktivasi hidden layer dan eksperimen intervensi papan.
  - **Analisis Kegagalan:** Kondisi dan jenis piece/pola papan di mana agen gagal.
  - **Diskusi & Keterbatasan:** Trade-off penyederhanaan game dan validitas eksternal.
  - **Kesimpulan:** Jawaban lugas atas pertanyaan riset.
  - **Referensi:** Sitasi literatur resmi.

### 3. `report/decisions.md` (Catatan Keputusan & Log Deviasi)
- **Waktu pengerjaan:** Diperbarui secara inkremental (*sambil jalan*) setiap kali ada keputusan arsitektur/teknis atau penyimpangan dari *Dokumen Proyek*.
- **Format per entri:**
  - Tanggal
  - Keputusan yang diambil
  - Alasan di balik keputusan
  - Alternatif yang dipertimbangkan dan ditolak
  - Dampak terhadap eksperimen / kode
- **Contoh isi wajib:** Hasil pilot tinggi papan, perubahan dari Dokumen Proyek, bug yang ditemukan dan cara perbaikannya.

### 4. Docstring dan Komentar Kode
- **Waktu pengerjaan:** Ditulis bersamaan dengan penulisan kode (bukan belakangan).
- **Aturan:**
  - Setiap fungsi/metode/kelas publik wajib memiliki docstring: tujuan, parameter/argumen, nilai kembalian, asumsi, dan alasan desain (*mengapa*, bukan sekadar *apa*).
  - Komentar inline hanya digunakan untuk bagian logika yang tidak langsung jelas dengan sendirinya.

---

## II. Dokumen yang Sangat Disarankan

### 5. `docs/environment.md` (Spesifikasi Teknis Environment)
- **Waktu pengerjaan:** Dibuat setelah Tahap 5 (implementasi antarmuka Gymnasium selesai).
- **Isi:**
  - Aturan game lengkap.
  - Definisi observasi secara presisi (dimensi tensor, tipe data, makna tiap kanal/elemen).
  - Ruang aksi diskrit dan mekanisme pendekodean aksi (rotasi, kolom).
  - Aturan action mask (pencegahan aksi keluar papan & penekanan rotasi duplikat).
  - Definisi fungsi reward pelatihan untuk setiap varian (R0 sampai R3).
  - Kondisi terminasi (`terminated`) dan pemotongan (`truncated`).
  - Struktur dictionary `info`.
  - Daftar simplifikasi game.

### 6. `docs/experiments.md` (Katalog Eksperimen)
- **Waktu pengerjaan:** Diisi bertahap seiring perancangan dan eksekusi tiap eksperimen.
- **Format tabel:**
  - ID Eksperimen (E1, E2, dst.)
  - Pertanyaan yang dijawab
  - File konfigurasi terkait (`configs/*.yaml`)
  - Perintah eksekusi di terminal
  - Jumlah seed yang dijalankan
  - Estimasi waktu komputasi
  - Lokasi artefak hasil / log

### 7. Komentar Terstruktur di `configs/*.yaml`
- **Waktu pengerjaan:** Ditulis saat file konfigurasi dibuat.
- **Aturan:**
  - Blok komentar pembuka di baris teratas: tujuan eksperimen, sel faktorial terkait (misal: *CEM + Fitur Linear* atau *TD + Raw Board*), serta perbedaan spesifik dibandingkan konfigurasi dasar (*base config*).

