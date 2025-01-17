# Pendeteksi TBC Berbasis Machine Learning

Proyek ini adalah aplikasi berbasis web untuk mendeteksi penyakit Tuberculosis (TBC) menggunakan model machine learning yang dikembangkan dengan TensorFlow. Aplikasi ini dilengkapi antarmuka pengguna yang modern dan responsif.

## Fitur Utama
- **Prediksi Gambar TBC**: Menggunakan dua model TensorFlow untuk validasi dan prediksi.
- **Preprocessing Otomatis**: Gambar diproses menggunakan OpenCV sebelum dianalisis.
- **UI Modern**: Antarmuka dibangun menggunakan TailwindCSS.
- **Privasi Data**: File pengguna tidak disimpan di server setelah diproses.

## Prasyarat
Pastikan Anda memiliki:
- Python 3.8 atau lebih baru.
- Git untuk mengkloning repositori.

## Cara Menggunakan

### 1. Clone Repositori
```bash
git clone https://github.com/username/pendeteksi-tbc.git
cd pendeteksi-tbc
```

### 2. Buat dan Aktifkan Virtual Environment
Di direktori proyek, jalankan:
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\Scripts\activate     # Untuk Windows
```

### 3. Instal Dependensi
Gunakan `pip` untuk menginstal semua dependensi:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
Jalankan server Flask dengan perintah:
```bash
python app.py
```
Aplikasi akan berjalan di `http://127.0.0.1:5000`

### 5. Gunakan Aplikasi
Buka browser dan akses `http://127.0.0.1:5000`. Unggah gambar X-ray untuk memulai prediksi
