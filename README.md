# Real-time Facial Emotion & Landmark Detection 

Aplikasi analisis wajah secara real-time yang mampu mendeteksi wajah, 68 titik kunci (facial landmarks), dan 7 emosi dasar (marah, jijik, takut, senang, netral, sedih, terkejut) menggunakan feed langsung dari webcam.

Proyek ini menggabungkan kecepatan deteksi objek dari **YOLOv8**, presisi deteksi landmark dari **dlib**, dan kekuatan analisis emosi dari **DeepFace** dalam satu alur kerja yang terintegrasi dan dioptimalkan untuk performa tinggi dengan akselerasi GPU.

---
 
## ✨ Fitur Utama

* **Deteksi Wajah Real-time**: Menggunakan model `YOLOv8n-face` yang cepat dan akurat untuk melokalisasi wajah dalam frame video.
* **Pengenalan Emosi**: Menganalisis ekspresi wajah untuk mengklasifikasikan 7 emosi dasar menggunakan library `DeepFace`.
* **Deteksi Landmark Wajah**: Mengidentifikasi dan menandai 68 titik kunci pada wajah (sudut mata, alis, hidung, bibir) menggunakan `dlib`.
* **Optimalisasi Performa**: Dioptimalkan untuk berjalan lancar dengan akselerasi GPU (NVIDIA CUDA) dan teknik frame skipping untuk mengurangi latensi.

---

## ⚙️ Cara Kerja

Aplikasi ini bekerja dalam sebuah pipeline yang berjalan secara terus-menerus untuk setiap frame dari webcam:

1. **Pengambilan Frame**: `OpenCV` menangkap frame video dari webcam.
2. **Deteksi Wajah (YOLOv8)**: Model `YOLOv8n-face` mendeteksi wajah dan mengembalikan bounding box.
3. **Analisis per Wajah**:

   * **Landmark (dlib)**: 68 titik kunci diprediksi pada area wajah.
   * **Emosi (DeepFace)**: Wajah dipotong lalu diklasifikasikan ke dalam salah satu dari 7 emosi dasar.
4. **Visualisasi**: Hasil analisis digambar kembali ke frame asli (bounding box, landmark, label emosi) dan ditampilkan di layar.

---

## 📂 Struktur Direktori

```
project-folder/
│
├── .venv/                   # Virtual environment Python
├── app_final.py             # Skrip utama aplikasi
├── requirements.txt         # Daftar library yang dibutuhkan
│
├── yolov8n-face-lindevs.pt  # Model YOLOv8 untuk deteksi wajah
└── shape_predictor_68_face_landmarks.dat  # Model dlib untuk landmark wajah
```

---

## 🔧 Instalasi

### Prasyarat

* Python 3.9+
* pip dan venv
* GPU NVIDIA dengan driver yang sesuai (disarankan untuk performa real-time)

### Langkah-langkah

#### 1. Clone Repositori

```bash
git clone https://github.com/NAMA_USER_ANDA/NAMA_REPOSITORI_ANDA.git
cd NAMA_REPOSITORI_ANDA
```

#### 2. Buat dan Aktifkan Virtual Environment

```bash
# Membuat virtual environment
python -m venv .venv

# Aktifkan (Windows PowerShell)
.\.venv\Scripts\activate
```

#### 3. Instal Dependensi

Pastikan PyTorch dengan dukungan CUDA diinstal.

```bash
# Instal PyTorch dengan dukungan CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instal dependensi lain
pip install -r requirements.txt
```

#### 4. Unduh File Model

* YOLOv8 Face: `yolov8n-face-lindevs.pt`
* dlib Landmark: `shape_predictor_68_face_landmarks.dat`

Tempatkan keduanya di direktori utama proyek.

---

## ▶️ Cara Menjalankan

Setelah instalasi selesai, jalankan aplikasi:

```bash
python app_final.py
```

Jendela webcam akan muncul dengan hasil deteksi wajah, landmark, dan emosi secara real-time.

Tekan tombol **`s`** untuk keluar dari aplikasi.

---

## 📌 Catatan

* Disarankan menggunakan GPU untuk performa optimal.
* `enforce_detection=False` pada DeepFace digunakan untuk mencegah deteksi ulang yang tidak perlu.

---

## 📜 Lisensi

Proyek ini bersifat open-source dan dapat dimodifikasi sesuai kebutuhan.
