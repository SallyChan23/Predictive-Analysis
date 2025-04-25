# Judul Proyek
Prediksi Nilai Matematika Siswa Menggunakan Machine Learning

# Identitas
**Nama**: Jeselyn Tania  
**Platform**: Kaggle Notebook  

# Pendahuluan
Kemampuan numerasi merupakan indikator penting dalam pendidikan. Nilai matematika menjadi salah satu tolok ukur utama untuk menilai prestasi akademik siswa. Banyak faktor yang dapat memengaruhi hasil nilai matematika, mulai dari kemampuan literasi hingga latar belakang sosial. Dengan menerapkan machine learning, kita dapat memprediksi nilai matematika siswa berdasarkan faktor-faktor tersebut dan memahami hubungan antar fitur yang memengaruhi performa akademik.

# Business Understanding
Proyek ini bertujuan untuk membangun model prediktif yang dapat membantu institusi pendidikan dalam:
- Mengidentifikasi siswa yang berpotensi memiliki nilai matematika rendah
- Mengambil tindakan intervensi lebih awal
- Memahami faktor apa yang paling memengaruhi nilai matematika siswa

# Data Understanding
Dataset digunakan dari Kaggle: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams), berisi data 1000 siswa dengan fitur:
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course
- reading score
- writing score
- math score (target)
---
Catatan tambahan:
Tidak terdapat missing value maupun duplikat pada dataset.

# Data Preparation
- **Encoding**: Semua fitur kategorikal (gender, race, dll) diubah menjadi numerik menggunakan LabelEncoder.
- **Feature Scaling**: Dilakukan StandardScaler pada fitur numerik untuk optimasi performa model berbasis regresi.
- **Train-Test Split**: Data dibagi menjadi 80% data pelatihan dan 20% data pengujian.

# Modeling
Dua model yang digunakan:
- **Linear Regression** sebagai baseline model.
- **Random Forest Regressor**, kemudian dilakukan **GridSearchCV** untuk tuning hyperparameter.

# Evaluation

| Model                 | MAE   | RMSE  | R²    |
|----------------------|--------|--------|-------|
| Linear Regression     | 4.13  | 5.32  | 0.88 |
| Random Forest         | 4.71  | 6.06  | 0.85 |
| Tuned Random Forest   | 4.69  | 6.04  | 0.85 |

Visualisasi hasil prediksi vs aktual menunjukkan sebagian besar titik mendekati garis ideal.  
Fitur `reading score` dan `writing score` memiliki feature importance tertinggi, menunjukkan literasi siswa sangat berpengaruh pada kemampuan numerik.

![image](https://github.com/user-attachments/assets/51c97a42-3c1f-4bf6-b3b5-066406123c71)

> Hasil scatter plot menunjukkan model dapat mempredeksi nilai matematika dengan cukup baik ditunjukkan oleh titik yang berada pada garis merah dan hanya beberapa titik saja yang diluar dari garis merah tersebut (ini bisa disebabkan oleh outliers).

![image](https://github.com/user-attachments/assets/e093f823-5d07-4fd6-8cd1-4634577a99cf)
> Dari hasil visualisasi tersebut, dapat ditunjukkan bahwa reading score serta writing score memiliki kontribusi tinggi dalam nilai matematika


# Kesimpulan
- Model machine learning mampu memprediksi nilai matematika dengan cukup baik (R² hingga 0.88).
- Fitur literasi (membaca dan menulis) adalah prediktor paling kuat.
- Hasil ini dapat digunakan sebagai dasar intervensi pendidikan berbasis data.

# Referensi
- https://www.kaggle.com/datasets/spscientist/students-performance-in-exams  
- Dokumentasi Scikit-Learn  
