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

---

## Solution Statement

Untuk mencapai hasil prediksi yang optimal, proyek ini menggunakan dua pendekatan model:

1. **Baseline Model: Linear Regression**  
   Model sederhana digunakan sebagai acuan awal performa prediksi.

2. **Random Forest Regressor dengan Hyperparameter Tuning**  
   Model ensembel yang memperbaiki performa prediksi dengan menggabungkan banyak pohon keputusan.

Semua model dievaluasi menggunakan metrik MAE, RMSE, dan R² Score untuk memilih model terbaik.


# Evaluation

| Model                 | MAE   | RMSE  | R²    |
|----------------------|--------|--------|-------|
| Linear Regression     | 4.13  | 5.32  | 0.88 |
| Random Forest         | 4.71  | 6.06  | 0.85 |
| Tuned Random Forest   | 4.69  | 6.04  | 0.85 |

> Dari kedua model ini, akan dipilih Random Forest algorithm, karena hasil evaluasi menunjukkan angka lebih baik dibanding Linear Regression
> Dari hasil tersebut dengan adanya tuning menjadi lebih baik, walau hanya sedikit. Dapat ditunjukkan dari hasil RMSE nya dari 6.06 menjadi 6.04

## Model Selection

Berdasarkan hasil evaluasi yang telah dilakukan, **Random Forest Regressor** dipilih sebagai model terbaik karena:

- **MAE** dan **RMSE** yang dihasilkan lebih rendah dibandingkan Linear Regression.
- **R² Score** dari Random Forest cukup tinggi dan stabil, menandakan model mampu menjelaskan variansi data dengan baik.
- Random Forest juga memiliki keunggulan dalam menangani hubungan non-linear antar fitur.

Dengan mempertimbangkan hasil evaluasi metrik, **Random Forest Regressor** menjadi model final yang digunakan dalam proyek ini.

**Hasil Visualisasi dengan Scatter Plot (Menggunakan Random Forest Setelah Tuning**

![image](https://github.com/user-attachments/assets/51c97a42-3c1f-4bf6-b3b5-066406123c71)

> Hasil scatter plot menunjukkan model dapat mempredeksi nilai matematika dengan cukup baik ditunjukkan oleh titik yang berada pada garis merah dan hanya beberapa titik saja yang diluar dari garis merah tersebut (ini bisa disebabkan oleh outliers).

![image](https://github.com/user-attachments/assets/e093f823-5d07-4fd6-8cd1-4634577a99cf)
> Dari hasil visualisasi tersebut, fitur `reading score` dan `writing score` memiliki feature importance tertinggi, menunjukkan literasi siswa sangat berpengaruh pada kemampuan numerik.

## Evaluation Metrics Explanation

Model dievaluasi menggunakan tiga metrik utama:

- **Mean Absolute Error (MAE):**  
  Mengukur rata-rata absolut selisih antara nilai aktual dan nilai prediksi.  
  Formula:  
  \[
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]  
  Semakin kecil MAE, semakin baik performa model.

- **Root Mean Squared Error (RMSE):**  
  Menghitung akar dari rata-rata kuadrat error. Lebih sensitif terhadap outlier dibandingkan MAE.  
  Formula:  
  \[
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  \]

- **R² Score (Coefficient of Determination):**  
  Mengukur seberapa baik variansi data dapat dijelaskan oleh model. Nilai R² berkisar antara 0 hingga 1, semakin mendekati 1 semakin baik.  
  Formula:  
  \[
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  \]

Ketiga metrik ini digunakan untuk memberikan gambaran menyeluruh mengenai akurasi dan keandalan model prediksi yang dibangun.


# Kesimpulan
- Model machine learning mampu memprediksi nilai matematika dengan cukup baik (R² hingga 0.88).
- Fitur literasi (membaca dan menulis) adalah prediktor paling kuat.
- Hasil ini dapat digunakan sebagai dasar intervensi pendidikan berbasis data.

# Referensi
- https://www.kaggle.com/datasets/spscientist/students-performance-in-exams  
- Dokumentasi Scikit-Learn  
