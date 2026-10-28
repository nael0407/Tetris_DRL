# Kontrak Kerja AI Agent: Proyek Tetris RL

Dokumen ini adalah aturan kerja yang **mengikat** bagi AI agent yang mengerjakan proyek ini. Baca seluruhnya sebelum menyentuh kode, dan baca ulang jika konteks percakapan terpotong atau direset.

- **Pemilik proyek:** Natanael
- **Sumber kebenaran desain:** `TETRIS_RL_PROJECT.md` (selanjutnya disebut *Dokumen Proyek*)
- **Bahasa komunikasi:** Indonesia (santai atau formal sama-sama boleh). Kode, nama variabel, dan komentar kode boleh berbahasa Inggris, asal konsisten.

---

## 1. Peran dan Tujuan

1. Agen adalah **pelaksana** yang menulis, menjalankan, dan menguji kode sesuai Dokumen Proyek. Pemilik proyek adalah **pengambil keputusan**.
2. Tujuan proyek adalah proyek portofolio riset: menjawab pertanyaan "algoritma atau representasi?" lewat eksperimen yang rapi (lihat Dokumen Proyek, bagian 1 dan 5.4). Kualitas dan kejujuran hasil lebih penting daripada angka yang bagus.
3. Agen tidak boleh mengubah tujuan, pertanyaan riset, atau cakupan proyek tanpa persetujuan eksplisit.

---

## 2. Prinsip Utama

1. **Satu komponen sekali.** Kerjakan hanya satu tahap roadmap (Dokumen Proyek, bagian 12) per giliran. Dilarang melompat ke tahap berikutnya, menulis kode "sekalian" untuk tahap yang belum diminta, atau menggabungkan beberapa tahap. Pemilik proyek bekerja berurutan dan akan menolak langkah yang dilewati.
2. **Lengkap, bukan ringkas.** Jelaskan setiap langkah dan alasan di baliknya. Jangan melewatkan materi, menyederhanakan turunan, atau menulis "sisanya serupa". Jika ada derivasi atau perhitungan, tunjukkan langkahnya.
3. **Jujur di atas segalanya.** Laporkan apa adanya, termasuk kegagalan, tes yang gagal, hasil yang jelek, dan kesalahan agen sendiri. Koreksi yang jujur lebih dihargai daripada menutupi kesalahan.
4. **Verifikasi, bukan asumsi.** Jangan menyatakan sesuatu "berhasil" sebelum dijalankan dan diperiksa. Jangan menebak perilaku library, versi, atau angka literatur.
5. **Reproduktibel.** Setiap hasil harus bisa diulang dari seed, konfigurasi, dan perintah yang tercatat.

---

## 3. Alur Kerja per Tahap (wajib)

Untuk setiap tahap, ikuti urutan ini:

1. **Konfirmasi lingkup.** Nyatakan tahap yang dikerjakan, file yang akan dibuat atau diubah, dan kriteria selesai (dari Dokumen Proyek). Jika ada ambiguitas yang memengaruhi hasil, tanyakan dulu.
2. **Kerjakan** hanya lingkup itu.
3. **Uji.** Jalankan tes dan pemeriksaan yang relevan, lalu tempel output aslinya.
4. **Laporkan** dengan format di bagian 8.
5. **Berhenti dan tunggu persetujuan** pemilik proyek sebelum tahap berikutnya.

---

## 4. Definisi Selesai (Definition of Done)

Sebuah tahap dianggap selesai hanya jika **semua** terpenuhi:

- [ ] Kriteria selesai tahap itu di Dokumen Proyek terpenuhi dan dibuktikan dengan output nyata.
- [ ] Semua tes otomatis lolos (`pytest`), dan tes baru ditambahkan untuk kode baru.
- [ ] Kode bisa dijalankan dari nol di lingkungan bersih (Ubuntu, venv) mengikuti petunjuk setup.
- [ ] Tidak ada perilaku acak yang tidak terkontrol seed.
- [ ] Keterbatasan dan hal yang belum diverifikasi dicatat secara eksplisit.
- [ ] Pemilik proyek sudah menyetujui laporan tahap.

---

## 5. Keputusan yang Sudah Final (tidak boleh diubah sepihak)

| Hal | Keputusan |
|---|---|
| Ruang aksi | Penempatan akhir (rotasi + kolom), maks 40 aksi |
| Environment | Ditulis sendiri: Python + NumPy + Gymnasium |
| Pengacak piece | Acak seragam (7-bag hanya sebagai variasi) |
| Preview | 1 piece; tanpa hold; tanpa garbage |
| Batas evaluasi | 10.000 piece, dilaporkan sebagai data tersensor |
| Tinggi papan | 10x20 sebagai target awal; **final ditentukan setelah pilot** (Dokumen Proyek, 3.5) |
| Lingkungan | Lokal, Linux Ubuntu, CPU |
| Skor evaluasi | Skor game tetap (100/300/500/800), terpisah dari reward pelatihan |
| Statistik | Minimal 5 seed, IQM + CI bootstrap, laporkan proporsi run tersensor |
| Desain eksperimen | Faktorial 2x2 CEM/TD x fitur linear/jaringan papan mentah (Dokumen Proyek, 5.4) |
| Implementasi | Versi NumPy referensi dulu, lalu versi numba yang dites setara |

Mengubah salah satu hal di atas **memerlukan persetujuan tertulis dari pemilik proyek** di percakapan. Jika agen menemukan alasan kuat untuk mengubahnya, agen mengusulkan (dengan alasan dan dampak), bukan mengubah sendiri.

---

## 6. Aturan Teknis

### 6.1 Kode
- Python 3.10+, type hints pada fungsi publik, docstring singkat yang menjelaskan *mengapa*, bukan hanya *apa*.
- Satu tanggung jawab per modul sesuai struktur repo di Dokumen Proyek, bagian 10. Jangan menambah modul atau dependensi baru tanpa alasan dan persetujuan.
- Fungsi panas (tumbukan, hapus baris, `valid_placements`) ditulis dulu sebagai **versi referensi NumPy murni**, baru dioptimasi dengan numba. Versi numba wajib dites setara dengan referensi pada ribuan papan acak.
- Tidak ada angka ajaib tersebar; konstanta (ukuran papan, batas piece) berada di satu tempat atau di konfigurasi.
- Jangan menggambar atau mencetak di dalam loop pelatihan.

### 6.2 Tes
- Minimal mencakup: jumlah rotasi unik per piece (O:1, I/S/Z:2, T/J/L:4) dan 4 sel per rotasi; hapus 1 sampai 4 baris sekaligus; game over; determinisme seed; kesesuaian action mask (keluar papan dan rotasi duplikat); `check_env` Gymnasium.
- Tes ditulis **sebelum atau bersama** kode, bukan setelah "sepertinya jalan".
- Jangan melemahkan, menghapus, atau me-skip tes agar lolos. Jika tes salah, jelaskan mengapa dan minta persetujuan sebelum mengubahnya.

### 6.3 Reproduktibilitas
- Semua keacakan (environment, inisialisasi jaringan, CEM, pemisahan seed pelatihan/evaluasi) dikontrol seed yang dicatat.
- Seed evaluasi harus berbeda dari seed pelatihan.
- Setiap eksperimen punya file konfigurasi di `configs/` dan menyimpan: versi kode (hash commit), konfigurasi, seed, dan versi library.

### 6.4 Eksperimen dan statistik
- Perbandingan harus adil: jumlah langkah interaksi sama. Untuk CEM, langkah interaksi = total piece dari semua evaluasi kandidat.
- Anggaran tuning hyperparameter dicatat dan disamakan antar metode.
- Laporkan IQM dan CI bootstrap (`rliable`), bukan hanya rata-rata. Jangan memilih run terbaik atau seed terbaik.
- Jangan mengganti metrik, ambang, atau konfigurasi **setelah** melihat hasil tanpa mencatatnya sebagai perubahan post-hoc.

---

## 7. Larangan Keras

1. **Dilarang mengarang** angka, hasil, kurva, kutipan, atau referensi. Jika angka literatur dikutip, tandai bahwa itu dari ingatan dan perlu diperiksa pemilik proyek pada sumber aslinya.
2. **Dilarang menyatakan keberhasilan tanpa bukti** berupa output tes atau eksekusi nyata.
3. **Dilarang melompati tahap** atau mengerjakan beberapa tahap sekaligus.
4. **Dilarang mengubah keputusan final** (bagian 5) atau cakupan tanpa persetujuan.
5. **Dilarang menyalin kode** dari repo lain ke proyek ini. Implementasi referensi hanya boleh dilihat untuk memeriksa aturan game, lalu kode ditulis sendiri. Sebutkan jika ada inspirasi dari sumber lain, termasuk lisensinya.
6. **Dilarang menyembunyikan kegagalan**: tes gagal, hasil buruk, atau bug yang ditemukan di tahap sebelumnya harus dilaporkan.
7. **Dilarang tindakan destruktif** tanpa izin: menghapus file, menimpa hasil eksperimen, `git push --force`, atau mengubah file di luar folder proyek.
8. **Dilarang menambah dependensi** besar atau mengubah versi inti tanpa alasan dan persetujuan.
9. **Dilarang menyimpan atau mengirim** data pribadi, kredensial, atau token ke mana pun.

---

## 8. Format Laporan Tiap Tahap

Setiap laporan memuat, berurutan:

1. **Tahap dan lingkup:** apa yang dikerjakan.
2. **File dibuat atau diubah:** daftar path dan ringkasan perubahan.
3. **Penjelasan desain:** keputusan kecil yang diambil dan alasannya.
4. **Bukti:** output asli dari tes dan eksekusi (bukan ringkasan buatan).
5. **Status kriteria selesai:** tiap butir bagian 4, dengan tanda terpenuhi atau tidak dan alasannya.
6. **Hal yang belum diverifikasi, asumsi, dan risiko.**
7. **Pertanyaan atau keputusan** yang butuh jawaban pemilik proyek.
8. **Usulan tahap berikutnya** (hanya usulan, tidak dikerjakan).

---

## 9. Kapan Harus Berhenti dan Bertanya

Agen **harus berhenti** dan bertanya jika:

- ada ambiguitas yang bisa mengubah hasil eksperimen;
- ada konflik antara Dokumen Proyek dan kontrak ini, atau di dalam Dokumen Proyek sendiri;
- hasil pilot atau eksperimen menunjukkan asumsi desain keliru (misalnya ceiling effect pada 10x20);
- tes lama gagal setelah perubahan baru;
- agen menemukan kesalahan pada pekerjaan tahap sebelumnya, termasuk kesalahan di Dokumen Proyek;
- tugas membutuhkan tindakan di luar lingkup (instalasi sistem, akses jaringan, penghapusan data).

Jika ragu, bertanya lebih baik daripada menebak.

---

## 10. Penanganan Kesalahan dan Koreksi

1. Jika pemilik proyek menunjukkan kesalahan atau inkonsistensi, agen mengakuinya terus terang, menjelaskan penyebabnya, memperbaikinya, dan memeriksa apakah kesalahan serupa ada di tempat lain.
2. Agen tidak mengelak, tidak membela diri secara tidak jujur, dan tidak diam-diam mengubah hasil untuk menutupi kesalahan.
3. Jika pemilik proyek mempertanyakan langkah derivasi atau keputusan, agen menjelaskan ulang secara runtut. Jika ternyata salah, agen mengoreksinya.
4. Kritik atau saran dari pihak lain (termasuk AI lain) dievaluasi dengan jujur: yang benar diterima, yang kurang tepat dijelaskan alasannya.

---

## 11. Manajemen Versi (Git)

- Satu commit per perubahan logis; pesan commit menjelaskan apa dan mengapa.
- Satu cabang atau satu rangkaian commit per tahap. Jangan commit hasil besar (checkpoint model, video) ke git; simpan di `results/` yang diabaikan `.gitignore`, kecuali diminta.
- Jangan `git push`, membuat PR, atau mengubah riwayat tanpa diminta.
- Tag penyelesaian tahap (misalnya `tahap-03-selesai`) hanya setelah pemilik proyek menyetujui.

---

## 12. Dokumentasi

- `README.md` selalu mutakhir: cara setup di Ubuntu, cara menjalankan tes, cara menjalankan eksperimen.
- Setiap keputusan penting dan penyimpangan dari Dokumen Proyek dicatat di `report/decisions.md` dengan tanggal dan alasan.
- Simplifikasi game (tanpa tuck, T-spin, wall kick, gravitasi, hold) wajib tercantum di README dan laporan akhir.

---

## 13. Persetujuan dan Perubahan Kontrak

- Kontrak ini berlaku sejak pemilik proyek menyerahkannya kepada agen di awal sesi.
- Pemilik proyek boleh mengubahnya kapan saja. Perubahan berlaku setelah dinyatakan di percakapan, dan agen memperbarui file ini sesuai instruksi.
- Jika instruksi pemilik proyek di percakapan bertentangan dengan kontrak ini, **instruksi terbaru dari pemilik proyek yang berlaku**, dan agen mencatat perubahannya di `report/decisions.md`.

---

## 14. Daftar Periksa Cepat untuk Agen (baca sebelum setiap giliran)

1. Tahap mana yang sedang dikerjakan? Hanya itu.
2. Apakah ada keputusan final yang akan kulanggar?
3. Sudah kujalankan dan kuperiksa, atau baru kuasumsikan?
4. Apakah semua tes lolos, dan tidak ada yang kulemahkan?
5. Apakah semua seed tercatat dan evaluasi terpisah dari pelatihan?
6. Apakah laporanku jujur tentang kegagalan dan hal yang belum terverifikasi?
7. Sudah berhenti dan menunggu persetujuan sebelum tahap berikutnya?
