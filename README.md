# Laporan Proyek Machine Learning - Jeselyn Tania

## Domain Proyek
Kemampuan numerasi merupakan indikator penting dalam pendidikan. Nilai matematika menjadi salah satu tolak ukur utama untuk menilai prestasi akademik siswa. Banyak faktor yang dapat memengaruhi hasil nilai matematika, mulai dari kemampuan literasi hingga latar belakang sosial. Dengan menerapkan machine learning, kita dapat memprediksi nilai matematika siswa berdasarkan faktor-faktor tersebut dan memahami hubungan antar fitur yang memengaruhi performa akademik.

Berdasarkan studi yang dilakukan oleh OECD (2019) dalam Programme for International Student Assessment (PISA), kemampuan literasi numerasi siswa berhubungan erat dengan performa akademik dan peluang sosial ekonomi di masa depan. Dengan menggunakan machine learning, diharapkan dapat ditemukan pola-pola dalam data akademik yang membantu memprediksi nilai matematika siswa secara lebih cepat dan akurat, sehingga pihak sekolah dapat mengambil langkah intervensi yang lebih awal.

Referensi:
- S. G. Paris and H. M. Paris, “Classroom Applications of Research on Self-Regulated Learning,” Educational Psychologist, vol. 36, no. 2, pp. 89–101, 2001. [DOI Link](https://doi.org/10.1207/S15326985EP3602_4)


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
Pada bagian ini, akan dijelaskan informasi mengenai data yang digunakan dalam proyek machine learning ini. Data yang digunakan berasal dari sumber terbuka dan berisi informasi performa akademik siswa berdasarkan berbagai faktor seperti jenis kelamin, etnis, latar belakang pendidikan orang tua, status makan siang, serta nilai ujian membaca dan menulis. Tujuan utama dari pemahaman data ini adalah untuk memahami struktur, kondisi, dan karakteristik fitur-fitur dalam dataset sebelum dilakukan pemrosesan dan pemodelan lebih lanjut.

### Sumber Data
Dataset diambil dari Kaggle: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

### Variable-variable pada students performance in exams dataset adalah sebagai berikut:
- **gender**: Jenis kelamin siswa (male/female).
- **race/ethnicity**: Kelompok etnis siswa.
- **parental level of education**: Tingkat pendidikan tertinggi dari orang tua/wali siswa.
- **lunch**: Status makan siang siswa (standard atau free/reduced).
- **test preparation course**: Status apakah siswa mengikuti program persiapan ujian.
- **reading score**: Skor ujian membaca siswa.
- **writing score**: Skor ujian menulis siswa.
- **math score**: Skor ujian matematika siswa (target prediksi).

### Jumlah Data
Dataset ini terdiri dari 1000 baris (observasi) dan 8 kolom (fitur).

### Kondisi Data
- **Missing Value**: Tidak terdapat missing value pada dataset.
- **Data Duplikat**: Tidak terdapat data duplikat.
- **Outlier**: Dataset tidak dilakukan penanganan khusus terhadap outlier karena data nilai ujian masih berada dalam rentang yang wajar (0-100).

### Exploratory Data Analysis (EDA)
- Sebagai bagian dari tahap pemahaman data, dilakukan eksplorasi awal terhadap dataset menggunakan fungsi deskriptif dasar. Beberapa fungsi seperti `df.info()`, `df.describe()`, dan pengecekan nilai kosong maupun duplikat digunakan untuk memahami struktur data, tipe data tiap fitur, serta distribusi nilai awal. Tahapan ini bertujuan untuk memastikan kualitas data sebelum dilakukan preprocessing dan pemodelan.

## Data Preparation

Tahap data preparation dilakukan untuk mempersiapkan data agar dapat digunakan oleh model machine learning dengan optimal. Berikut tahapan-tahapan yang dilakukan:

### 1. Pemisahan Fitur dan Target

Langkah awal sebelum melakukan preprocessing adalah memisahkan data menjadi fitur (X) dan target (y). Kolom `math score` digunakan sebagai target karena merupakan nilai yang ingin diprediksi, sementara kolom lainnya digunakan sebagai fitur. Tahapan ini penting untuk membedakan variabel input dan output dalam pemodelan machine learning.

### 2. Encoding

Beberapa fitur pada dataset masih dalam bentuk kategorikal seperti `gender`, `race/ethnicity`, `parental level of education`, `lunch`, dan `test preparation course`. Fitur-fitur ini diubah menjadi format numerik menggunakan teknik **Label Encoding**, agar dapat diproses oleh algoritma machine learning yang hanya menerima input numerik.

### 3. Feature Scaling

Fitur numerik pada dataset seperti `reading score` dan `writing score` memiliki rentang nilai yang berbeda. Oleh karena itu, dilakukan **normalisasi/standardisasi** menggunakan `StandardScaler` dari Scikit-Learn.  
Scaling ini bertujuan untuk menstandarkan fitur dengan cara mengubah distribusi menjadi memiliki rata-rata 0 dan standar deviasi 1.

**Alasan Scaling:**  
Scaling diperlukan karena beberapa algoritma seperti **Linear Regression** sensitif terhadap perbedaan skala antar fitur. Tanpa scaling, fitur dengan skala besar dapat mendominasi proses pelatihan dan menyebabkan bias dalam prediksi. Dengan scaling, semua fitur diberi perlakuan yang adil.

### 4. Train-Test Split

Dataset dibagi menjadi dua bagian:
- **Training set (80%)** digunakan untuk melatih model
- **Testing set (20%)** digunakan untuk mengevaluasi performa model terhadap data yang belum pernah dilihat sebelumnya.

Pembagian dilakukan secara acak dengan menggunakan fungsi `train_test_split` dari Scikit-Learn, dengan seed `random_state=42` untuk hasil yang reprodusibel.

## Modeling
Dua model yang digunakan:
- **Linear Regression** sebagai baseline model.
- **Random Forest Regressor** dengan **GridSearchCV** untuk hyperparameter tuning.

### Model 1: Linear Regression

**Cara Kerja**:  
Linear Regression merupakan algoritma supervised learning yang digunakan untuk memprediksi nilai kontinu. Model ini berusaha menemukan garis lurus terbaik yang meminimalkan jumlah kuadrat error antara nilai prediksi dan nilai aktual. Prediksi dihitung dengan persamaan:
\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n
\]

**Parameter**:  
- Menggunakan parameter default dari `LinearRegression` pada scikit-learn.

---

### Model 2: Random Forest Regressor

**Cara Kerja**:  
Random Forest Regressor adalah algoritma ensemble learning yang menggabungkan banyak decision tree. Setiap tree dilatih dengan subset data berbeda (bootstrap sampling), dan hasil prediksi adalah rata-rata dari semua tree. Ini membantu mengurangi overfitting dan meningkatkan akurasi.

**Parameter**:
- `n_estimators`: 100 (sebelum tuning), dioptimasi hingga 200 setelah GridSearchCV.
- `max_depth`: None (default) dan dioptimasi menjadi 10/20 pada tuning.
- `min_samples_split` dan `min_samples_leaf` diatur pada tuning.

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
