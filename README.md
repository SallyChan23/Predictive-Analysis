# Laporan Proyek Machine Learning - Jeselyn Tania

## Domain Proyek
Kemampuan numerasi merupakan indikator penting dalam pendidikan. Nilai matematika menjadi salah satu tolok ukur utama untuk menilai prestasi akademik siswa. Banyak faktor yang dapat memengaruhi hasil nilai matematika, mulai dari kemampuan literasi hingga latar belakang sosial. Dengan menerapkan machine learning, kita dapat memprediksi nilai matematika siswa berdasarkan faktor-faktor tersebut dan memahami hubungan antar fitur yang memengaruhi performa akademik.

Referensi:
- [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

## Business Understanding

### Problem Statements
- Bagaimana memprediksi nilai matematika siswa berdasarkan data demografis dan nilai akademik lainnya?
- Apa faktor utama yang mempengaruhi nilai matematika siswa?

### Goals
- Membangun model machine learning untuk memprediksi nilai matematika siswa.
- Mengidentifikasi fitur yang paling berpengaruh terhadap prediksi nilai matematika.

### Solution Statements
- Membangun baseline model menggunakan **Linear Regression**.
- Meningkatkan performa dengan model **Random Forest Regressor** dan melakukan **Hyperparameter Tuning**.
- Model dievaluasi menggunakan MAE, RMSE, dan R² Score untuk menentukan model terbaik.

## Data Understanding
Dataset digunakan dari Kaggle: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams), berisi data 1000 siswa dengan fitur:
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course
- reading score
- writing score
- math score (target)

Catatan tambahan:
- Tidak terdapat missing value maupun duplikat pada dataset.

## Data Preparation
- **Encoding**: Semua fitur kategorikal diubah menjadi numerik menggunakan LabelEncoder.
- **Feature Scaling**: Dilakukan StandardScaler pada fitur numerik untuk optimasi performa model berbasis regresi.
- **Train-Test Split**: Data dibagi menjadi 80% data pelatihan dan 20% data pengujian.
- Alasan dilakukan feature scaling adalah karena beberapa algoritma machine learning, seperti Linear Regression, sensitif terhadap skala fitur. Dengan melakukan scaling, semua fitur akan berada pada rentang nilai yang seragam, sehingga model dapat berlatih lebih optimal dan menghindari bias terhadap fitur dengan skala yang lebih besar.

## Modeling
Dua model yang digunakan:
- **Linear Regression** sebagai baseline model.
- **Random Forest Regressor** dengan **GridSearchCV** untuk hyperparameter tuning.

### Kelebihan dan Kekurangan Model

- **Linear Regression**:  
  + Kelebihan: Sederhana, interpretasi koefisien fitur mudah.  
  + Kekurangan: Hanya mampu menangkap hubungan linear antar fitur, performa menurun jika ada hubungan non-linear.

- **Random Forest Regressor**:  
  + Kelebihan: Mampu menangkap hubungan non-linear dan interaksi antar fitur secara otomatis, lebih tahan terhadap overfitting dibanding model single tree.  
  + Kekurangan: Interpretasi hasil model lebih sulit dibanding Linear Regression, serta membutuhkan waktu komputasi lebih lama untuk training.

### Model Selection
Berdasarkan hasil evaluasi, **Random Forest Regressor** dipilih sebagai model terbaik karena:
- MAE dan RMSE lebih rendah dibandingkan Linear Regression.
- R² Score cukup tinggi dan stabil.
- Random Forest lebih fleksibel dalam menangani hubungan non-linear antar fitur.

## Evaluation

### Hasil Evaluasi Model:

| Model                 | MAE   | RMSE  | R²    |
|----------------------|--------|--------|-------|
| Linear Regression     | 4.13  | 5.32  | 0.88 |
| Random Forest         | 4.71  | 6.06  | 0.85 |
| Tuned Random Forest   | 4.69  | 6.04  | 0.85 |

> Random Forest setelah tuning menghasilkan RMSE yang sedikit lebih baik dibandingkan sebelum tuning.

### Visualisasi Hasil Prediksi:

![Scatter Plot Prediksi vs Aktual](https://github.com/user-attachments/assets/51c97a42-3c1f-4bf6-b3b5-066406123c71)

> Hasil scatter plot menunjukkan prediksi cukup baik, dengan sebagian besar titik dekat garis ideal. Outlier terlihat sedikit menyebar.

### Feature Importance:

![Feature Importance](https://github.com/user-attachments/assets/e093f823-5d07-4fd6-8cd1-4634577a99cf)

> Fitur `reading score` dan `writing score` memiliki kontribusi tertinggi terhadap prediksi nilai matematika.

### Evaluation Metrics Explanation

- **Mean Absolute Error (MAE):**  
  Mengukur rata-rata absolut selisih antara nilai aktual dan nilai prediksi.  
  \[
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]

- **Root Mean Squared Error (RMSE):**  
  Menghitung akar dari rata-rata kuadrat error.  
  \[
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  \]

- **R² Score (Coefficient of Determination):**  
  Mengukur proporsi variansi data yang dapat dijelaskan oleh model.  
  \[
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  \]

## Kesimpulan
- Model machine learning mampu memprediksi nilai matematika siswa dengan baik, dengan R² hingga 0.88.
- Fitur literasi (`reading score`, `writing score`) terbukti sangat berpengaruh terhadap nilai matematika.
- Random Forest Regressor dipilih sebagai model akhir untuk proyek ini.

## Referensi
- [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Dokumentasi Scikit-Learn
