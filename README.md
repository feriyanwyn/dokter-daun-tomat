
# 🍅 Dokter Daun Tomat – CNN Pendeteksi Penyakit Tanaman

**Dokter Daun Tomat** adalah proyek Machine Learning berbasis Convolutional Neural Network (CNN) untuk mendeteksi penyakit pada daun tanaman tomat melalui gambar. Sistem menggunakan arsitektur **MobileNetV2 dengan Transfer Learning** untuk mengklasifikasikan 10 kondisi daun tomat, terdiri dari 9 jenis penyakit dan 1 kondisi sehat.

Proyek ini dikembangkan menggunakan Python dan TensorFlow/Keras melalui Google Colab.

## 📌 Fitur Utama

- **Deteksi Penyakit Daun:** Mengidentifikasi 9 jenis penyakit dan daun tomat sehat.
- **Transfer Learning:** Menggunakan MobileNetV2 pretrained ImageNet.
- **Image Classification:** Mengklasifikasikan gambar daun tomat ke dalam 10 kelas.
- **Grad-CAM:** Memvisualisasikan bagian gambar yang menjadi dasar keputusan model.
- **Upload Gambar:** Pengguna dapat mengunggah foto daun tomat untuk mendapatkan hasil diagnosis.
- **Rekomendasi Penanganan:** Menampilkan informasi penyakit dan tips penanganan awal dalam Bahasa Indonesia.
- **Model Export:** Mendukung ekspor model ke format TensorFlow Lite (.tflite) untuk integrasi aplikasi Android/iOS.

## 🦠 Kelas Penyakit yang Dideteksi

Model dilatih untuk mengenali 10 kondisi daun tomat berikut:

| No. | Nama Penyakit | Keterangan |
|---|---|---|
| 1 | Bacterial Spot | Bercak bakteri |
| 2 | Early Blight | Hawar daun awal |
| 3 | Late Blight | Hawar daun akhir / busuk daun |
| 4 | Leaf Mold | Jamur daun |
| 5 | Septoria | Bercak daun Septoria |
| 6 | Spider Mites | Serangan tungau laba-laba |
| 7 | Target Spot | Bercak target |
| 8 | Yellow Leaf Curl | Virus keriting kuning (TYLCV) |
| 9 | Mosaic | Virus mosaik |
| 10 | Healthy | Daun sehat |

## 📂 Dataset

Dataset yang digunakan adalah **PlantVillage Dataset** yang diperoleh melalui TensorFlow Datasets (TFDS).

Sumber dataset:
- [PlantVillage – TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)

Informasi dataset:
- Sumber: PlantVillage.
- Pengambilan data: Otomatis melalui TensorFlow Datasets.
- Kategori: 10 kelas daun tomat.
- Ukuran gambar: 160 × 160 piksel.
- Format gambar: RGB.
- Dataset diproses dengan mengambil hanya gambar yang termasuk kategori tanaman tomat.

Notebook juga menyediakan fitur opsional untuk menyimpan dataset yang telah diproses ke dalam file ZIP berdasarkan kategori penyakit.

## ⚙️ Metodologi

Tahapan pengembangan model meliputi:

1. **Import Library** – Mengimpor TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, dan TensorFlow Datasets.
2. **Dataset Preparation** – Mengunduh dataset PlantVillage dan mengambil 10 kelas daun tomat.
3. **Image Preprocessing** – Mengubah ukuran gambar menjadi 160 × 160 piksel dengan format RGB.
4. **Data Splitting** – Membagi dataset menggunakan Stratified Sampling menjadi data training, validation, dan testing.
5. **Data Augmentation** – Melakukan transformasi gambar berupa horizontal flip, vertical flip, brightness, dan contrast.
6. **Class Weight** – Menggunakan pembobotan kelas untuk membantu menangani ketidakseimbangan jumlah data.
7. **Model Development** – Membangun model CNN menggunakan MobileNetV2 pretrained ImageNet.
8. **Model Training** – Melatih model melalui dua tahap, yaitu training classification head dan fine-tuning.
9. **Model Evaluation** – Mengukur performa model menggunakan akurasi, classification report, dan confusion matrix.
10. **Grad-CAM Visualization** – Menampilkan area gambar yang memengaruhi hasil prediksi.
11. **Image Diagnosis** – Menguji model menggunakan gambar daun tomat yang diunggah pengguna.
12. **Model Export** – Menyimpan model dalam format Keras dan TensorFlow Lite.

## 🧠 Arsitektur Model

Model menggunakan **MobileNetV2** sebagai feature extractor dengan bobot pretrained ImageNet.

Arsitektur model:

```text
Input Image (160 × 160 × 3)
          |
          v
Image Rescaling (0-255 ke -1 hingga 1)
          |
          v
MobileNetV2 (Pretrained ImageNet)
          |
          v
Global Average Pooling 2D
          |
          v
Dropout (0.25)
          |
          v
Dense (10, Softmax)
          |
          v
Classification Output
          |
          v
10 Kelas Daun Tomat
```

### Tahapan Training

Training dilakukan melalui dua tahap:

**Tahap 1 – Transfer Learning**
- Menggunakan MobileNetV2 pretrained ImageNet.
- Layer utama MobileNetV2 dibekukan (*frozen*).
- Melatih classification head.
- Optimizer: Adam.
- Learning rate: 0.001.
- Maksimal epoch: 6.

**Tahap 2 – Fine-Tuning**
- Membuka 40 layer terakhir MobileNetV2.
- Layer Batch Normalization tetap dibekukan.
- Melatih kembali model dengan learning rate lebih kecil.
- Optimizer: Adam.
- Learning rate: 0.00003.
- Maksimal epoch: 10.

Kedua tahap menggunakan Early Stopping dengan patience 3 dan mengembalikan bobot terbaik berdasarkan validation loss.

## 📊 Evaluasi Model

Evaluasi dilakukan menggunakan data testing yang tidak digunakan selama proses training.

Metrik evaluasi yang digunakan:

| Metrik | Fungsi |
|---|---|
| Accuracy | Mengukur persentase prediksi yang benar. |
| Precision | Mengukur ketepatan hasil prediksi pada setiap kelas. |
| Recall | Mengukur kemampuan model mengenali seluruh data pada setiap kelas. |
| F1-Score | Mengukur keseimbangan antara precision dan recall. |
| Confusion Matrix | Menampilkan perbandingan hasil prediksi dengan label sebenarnya. |

Notebook menghasilkan:
- Grafik akurasi training dan validation.
- Grafik loss training dan validation.
- Classification report.
- Confusion matrix.
- Visualisasi hasil prediksi yang benar dan salah.

Nilai performa akhir diperoleh setelah notebook dijalankan melalui proses training dan evaluasi.

## 🔍 Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) digunakan untuk mengetahui bagian gambar yang paling memengaruhi keputusan model.

Visualisasi menggunakan heatmap:
- Merah/Kuning: Area yang memiliki pengaruh lebih besar terhadap prediksi.
- Biru: Area yang memiliki pengaruh lebih rendah.

Fitur ini membantu memahami apakah model memperhatikan bagian daun yang mengalami bercak atau kerusakan ketika melakukan klasifikasi.

## 📷 Diagnosis Gambar Daun Tomat

Pengguna dapat mengunggah gambar daun tomat untuk mendapatkan hasil prediksi.

Sistem akan menampilkan:
1. Gambar daun yang diunggah.
2. Visualisasi Grad-CAM.
3. Tiga prediksi dengan probabilitas tertinggi.
4. Tingkat keyakinan model dalam persentase.
5. Informasi penyakit.
6. Rekomendasi penanganan awal.

Jika tingkat keyakinan prediksi tertinggi kurang dari 70%, sistem memberikan peringatan bahwa model kurang yakin terhadap hasil diagnosis.

**Tips mengambil gambar:**
- Gunakan satu daun dalam satu foto.
- Ambil gambar dari jarak dekat.
- Pastikan pencahayaan cukup.
- Gunakan latar belakang polos.

## 📱 Ekspor Model ke TensorFlow Lite

Model dapat diekspor ke format TensorFlow Lite agar lebih mudah diintegrasikan ke aplikasi mobile.

File yang dihasilkan:

| File | Keterangan |
|---|---|
| `daun_tomat.tflite` | Model TensorFlow Lite untuk aplikasi mobile. |
| `labels.txt` | Daftar 10 kelas output model. |
| `daun_tomat.keras` | Model dalam format Keras. |

Spesifikasi input model:
- Ukuran gambar: 160 × 160 piksel.
- Channel: 3 (RGB).
- Tipe data input: Float32.
- Rentang input: 0–255.
- Output: 10 probabilitas klasifikasi sesuai urutan `labels.txt`.

Model TensorFlow Lite menggunakan optimasi default dari TensorFlow Lite Converter.

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa pemrograman utama. |
| Google Colab | Lingkungan pengembangan dan training. |
| TensorFlow | Framework Machine Learning. |
| Keras | Pembangunan dan pelatihan model CNN. |
| MobileNetV2 | Arsitektur CNN pretrained. |
| TensorFlow Datasets | Pengambilan dataset PlantVillage. |
| NumPy | Pengolahan data numerik. |
| Matplotlib | Visualisasi gambar dan grafik. |
| Seaborn | Visualisasi confusion matrix. |
| Scikit-learn | Data splitting dan evaluasi model. |
| Pillow (PIL) | Pemrosesan gambar. |

## 🚀 Cara Menjalankan

### Menggunakan Google Colab

1. Clone atau unduh repository GitHub.
2. Buka [Google Colab](https://colab.research.google.com/).
3. Pilih menu **File → Upload notebook**.
4. Upload file `CNN_Dokter_Daun_Tomat_Feriyan_50423500.ipynb`.
5. Pilih **Runtime → Change runtime type**.
6. Pilih GPU sebagai hardware accelerator (disarankan NVIDIA T4).
7. Jalankan notebook menggunakan **Run all**.
8. Tunggu proses instalasi library, pengunduhan dataset, preprocessing, dan training selesai.

Estimasi waktu pengerjaan sekitar 15–25 menit, bergantung pada ketersediaan GPU dan kecepatan internet.

### Menggunakan Model TensorFlow Lite

Setelah menjalankan proses ekspor model, gunakan file berikut untuk integrasi aplikasi mobile:

- `daun_tomat.tflite`
- `labels.txt`

Pastikan preprocessing gambar pada aplikasi mengikuti spesifikasi input model.

## 📁 Struktur Project

```text
CNN-Dokter-Daun-Tomat/
│
├── CNN_Dokter_Daun_Tomat_Feriyan_50423500.ipynb
│
├── daun_tomat.tflite
│
├── daun_tomat.keras
│
├── labels.txt
│
└── dataset_daun_tomat/
    ├── bacterial_spot/
    ├── early_blight/
    ├── late_blight/
    ├── leaf_mold/
    ├── septoria/
    ├── spider_mites/
    ├── target_spot/
    ├── yellow_leaf_curl/
    ├── mosaic/
    └── healthy/
```

File dataset dan model hasil ekspor akan dibuat ketika bagian terkait dijalankan di Google Colab.

## ⚠️ Batasan Sistem

- Model hanya dirancang untuk mengenali 10 kondisi daun tomat yang terdapat dalam dataset.
- Model dapat tetap memberikan prediksi ketika gambar yang diunggah bukan daun tomat.
- Hasil prediksi merupakan estimasi berdasarkan pola visual pada dataset.
- Kondisi pencahayaan, kualitas gambar, sudut pengambilan, dan latar belakang dapat memengaruhi hasil prediksi.
- Rekomendasi penanganan yang diberikan bersifat umum dan bukan pengganti diagnosis ahli pertanian.

## 👨‍💻 Developer

**Feriyan Eka Nanda**

Informatics Student | Machine Learning & Artificial Intelligence

GitHub: [@feriyanwyn](https://github.com/feriyanwyn)

---

**Disclaimer:** Proyek ini dikembangkan untuk tujuan pembelajaran, eksperimen Machine Learning, dan pengenalan penyakit tanaman menggunakan computer vision. Hasil diagnosis tidak menggantikan pemeriksaan langsung oleh ahli atau penyuluh pertanian.
