# Prediksi Nilai Matematika Siswa Menggunakan Traditional Machine Learning

**Nama**: Jeselyn Tania
**Platform**: Kaggle Notebook  
---

## 1. Latar Belakang

Kemampuan numerasi merupakan indikator penting dalam pendidikan. Dalam konteks pembelajaran siswa, nilai matematika menjadi salah satu tolak ukur yang dapat digunakan untuk menilai prestasi akademik. Namun demikian, banyak faktor yang dapat memengaruhi hasil nilai matematika siswa, mulai dari kemampuan literasi hingga latar belakang sosial.

Dengan memanfaatkan machine learning, kita dapat memprediksi nilai matematika siswa berdasarkan faktor-faktor tersebut. Proyek ini bertujuan untuk memahami pola tersebut dan mengevaluasi seberapa baik model machine learning dapat melakukan prediksi berbasis data pendidikan.

---

## 2. Permasalahan

Bagaimana memprediksi nilai matematika siswa berdasarkan fitur-fitur seperti nilai membaca, nilai menulis, gender, ras, tingkat pendidikan orang tua, dan keikutsertaan dalam program persiapan ujian?

---

## 3. Tujuan Proyek

- Membangun model machine learning untuk memprediksi nilai matematika siswa.
- Mengevaluasi performa model menggunakan metrik MAE, RMSE, dan R², serta menunjukkan nya dengan scatter plot (apakah model tersebut sudah akurat dalam memprediksi atau belum)
- Mengidentifikasi fitur yang paling berpengaruh terhadap prediksi nilai matematika.

---

## 4. Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari Kaggle:  
[Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

- Jumlah data: 1000 baris
- Fitur: gender, race/ethnicity, parental level of education, lunch, test preparation course, reading score, writing score
- Target prediksi: `math score`
- Semua data sudah bersih: tidak terdapat nilai kosong maupun duplikat.

---

## 5. Proses Persiapan Data

Sebelum membangun model, dilakukan beberapa tahap persiapan data sebagai berikut:

- **Encoding Data Kategorikal**  
  Kolom-kolom seperti `gender`, `race/ethnicity`, `parental level of education`, `lunch`, dan `test preparation course` diubah menjadi nilai numerik menggunakan `LabelEncoder`.

- **Feature Scaling**  
  Untuk memastikan performa optimal pada model seperti Linear Regression, fitur-fitur numerik di-scale menggunakan `StandardScaler`.

- **Train-Test Split**  
  Dataset dibagi menjadi data pelatihan dan data pengujian dengan rasio 80:20. Tujuannya untuk mengevaluasi model secara adil.

---

## 6. Modeling dan Evaluasi

### Model yang Digunakan:
1. **Linear Regression**  
   Model dasar yang digunakan sebagai benchmark awal.
2. **Random Forest Regressor**  
   Model non-linear yang lebih kompleks, dilatih dan dioptimasi menggunakan hyperparameter tuning.

### Hasil Evaluasi:

| Model                 | MAE   | RMSE  | R²    |
|----------------------|--------|--------|-------|
| Linear Regression     | 4.13  | 5.32  | 0.88 |
| Random Forest         | 4.71  | 6.06  | 0.85 |
| Tuned Random Forest   | 4.69  | 6.04  | 0.85 |

> Model terbaik berdasarkan nilai error terendah adalah **Linear Regression**, meskipun model Random Forest telah dituning dengan GridSearchCV.

Evaluasi dilakukan menggunakan tiga metrik:
- **MAE (Mean Absolute Error)**: rata-rata selisih absolut antara prediksi dan data aktual.
- **RMSE (Root Mean Squared Error)**: penalti lebih tinggi untuk error besar.
- **R² Score**: seberapa besar variasi nilai target yang dapat dijelaskan oleh model.

---
## 7. Visualisasi Hasil Prediksi

Untuk melihat seberapa baik model memprediksi nilai matematika, dibuat visualisasi hasil prediksi terhadap nilai aktual pada data uji.

![image](https://github.com/user-attachments/assets/6117b124-72ff-4428-955f-2ca3e13b87d7)

> Grafik di atas menunjukkan bahwa sebagian besar prediksi mendekati garis ideal (prediksi = aktual). Ini mengindikasikan bahwa model mampu menangkap pola dengan cukup akurat. Beberapa outlier terlihat menyimpang, namun secara umum sebaran prediksi cukup konsisten.

---

## 8. Feature Importance

Analisis feature importance dilakukan menggunakan model **Random Forest** untuk melihat fitur mana yang paling berpengaruh dalam memprediksi nilai matematika.

Hasilnya:
![image](https://github.com/user-attachments/assets/36985e1a-1289-45f9-9884-d5d6ced68abd)

> Fitur `reading score` memiliki kontribusi terbesar dalam memprediksi nilai matematika, diikuti oleh `writing score`. Hal ini mengindikasikan bahwa kemampuan literasi siswa (membaca dan menulis) sangat berkorelasi dengan kemampuan numerik mereka. Faktor lain seperti `gender`, `race/ethnicity`, dan `parental education` memberikan pengaruh yang lebih kecil.

---
## 9. Kesimpulan

Proyek ini berhasil membangun model machine learning untuk memprediksi nilai matematika siswa berdasarkan data demografis dan nilai akademik lainnya. Hasil evaluasi menunjukkan bahwa:

- **Linear Regression** memberikan performa terbaik dengan R² sebesar 0.88.
- Fitur yang paling berpengaruh terhadap nilai matematika adalah **reading score** dan **writing score**, menunjukkan bahwa kemampuan literasi sangat berkaitan erat dengan kemampuan numerik.
- Visualisasi hasil prediksi menunjukkan bahwa sebagian besar prediksi cukup akurat, dengan penyimpangan yang minim.

Dengan pendekatan sederhana, model mampu memberikan insight yang kuat dalam konteks pendidikan, dan bisa dikembangkan lebih lanjut untuk membantu analisis performa siswa secara prediktif.

---

## 10. Penutup

Proyek ini menunjukkan potensi machine learning untuk diterapkan dalam bidang pendidikan, khususnya dalam menganalisis dan memprediksi performa siswa. Dengan data yang tepat dan proses analisis yang sistematis, kita dapat menghasilkan model yang tidak hanya akurat, namun juga memberikan insight yang bermanfaat untuk pengambilan keputusan.

Seluruh proses mulai dari eksplorasi data, preprocessing, pemodelan, evaluasi, hingga visualisasi telah didokumentasikan dalam file `.ipynb`, `.py`, dan laporan ini (`.md`). Semoga proyek ini dapat menjadi kontribusi kecil dalam menggabungkan teknologi dan pendidikan.
