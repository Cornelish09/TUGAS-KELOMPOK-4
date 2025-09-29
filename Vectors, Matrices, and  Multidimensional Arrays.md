# Vectors, Matrices, and  Multidimensional Arrays

## Pendahuluan


### Tujuan Tugas

Tugas ini bertujuan untuk membedah secara mendalam sintaks dan konsep fundamental dari pustaka (library) NumPy, dengan berpedoman pada materi yang disajikan dalam Bab 2 buku *Numerical Python*. Tugas ini dirancang sebagai panduan komprehensif untuk membangun kemahiran dalam komputasi numerik dan pengolahan data, yang merupakan pilar utama dalam ilmu data dan rekayasa modern.

### Mengapa NumPy?

Dalam ekosistem komputasi ilmiah Python, NumPy memegang peranan sentral yang tak tergantikan. Ia menyediakan objek array multidimensi, ndarray, yang menjadi fondasi bagi hampir semua pustaka lain di bidang ini, termasuk SciPy, Pandas, dan Scikit-learn. Dibandingkan dengan struktur data bawaan Python seperti list, ndarray menawarkan keunggulan signifikan untuk data numerik. list Python adalah wadah heterogen yang fleksibel, tetapi fleksibilitas ini harus dibayar dengan kinerja yang lebih lambat karena setiap elemen disimpan sebagai objek Python terpisah yang bisa tersebar di memori. Sebaliknya, ndarray adalah wadah homogen (semua elemen memiliki tipe data yang sama) yang menyimpan data dalam satu blok memori yang berdekatan (contiguous).

### Konsep Kunci: Vektorisasi

Paradigma fundamental yang mendasari kekuatan NumPy adalah vektorisasi. Alih-alih menulis *loop* eksplisit untuk mengiterasi setiap elemen array, sebuah proses yang lambat dalam Python vektorisasi memungkinkan kita untuk menerapkan operasi pada seluruh array sekaligus. Operasi-operasi ini (misalnya, penjumlahan atau perkalian array) dieksekusi di balik layar oleh kode C atau Fortran yang sangat dioptimalkan. Hasilnya adalah kode yang tidak hanya jauh lebih cepat, tetapi juga lebih ringkas, lebih mudah dibaca, dan lebih mirip dengan notasi matematika standar. Konsep ini akan menjadi tema yang berulang dan kunci untuk memahami efisiensi di seluruh laporan ini.

## 1. Mengenali Karakteristik Data (Objek ndarray)


### Fondasi NumPy: Objek ndarray

Inti dari pustaka NumPy adalah objek ndarray, yang merupakan singkatan dari "N-dimensional array". Ini adalah sebuah wadah data multidimensi yang memiliki dua karakteristik utama yang membedakannya dari list Python:
Homogen: Semua elemen di dalam ndarray harus memiliki tipe data yang sama. Keterbatasan ini adalah sumber kekuatannya, karena memungkinkan data disimpan secara padat dan efisien di memori.
Ukuran Tetap (Fixed-size): Setelah sebuah ndarray dibuat, ukurannya tidak dapat diubah. Operasi yang tampaknya mengubah ukuran, seperti menambahkan elemen, sebenarnya membuat array baru dan menyalin data lama ke dalamnya.
Struktur yang kaku ini memungkinkan NumPy untuk melakukan optimisasi tingkat rendah yang tidak mungkin dilakukan pada list Python yang dinamis.

### Metadata: Atribut-Atribut Array

Setiap instance ndarray dilengkapi dengan metadata penting yang mendeskripsikan struktur dan kontennya. Atribut-atribut ini bukan sekadar informasi, melainkan instruksi vital yang digunakan oleh *engine* C dan Fortran di balik layar NumPy untuk menavigasi blok memori secara efisien. Memahami atribut-atribut ini adalah langkah pertama untuk menguasai NumPy.
ndim: Menyatakan jumlah dimensi atau sumbu (*axes*) dari array. Sebuah vektor memiliki ndim=1, matriks memiliki ndim=2, dan seterusnya.
shape: Sebuah *tuple* Python yang berisi jumlah elemen untuk setiap dimensi. Misalnya, sebuah matriks dengan 3 baris dan 2 kolom akan memiliki shape (3, 2).
size: Menyatakan jumlah total elemen dalam array. Nilai ini adalah hasil perkalian dari semua angka dalam shape. Untuk array dengan shape (3, 2), size-nya adalah 3×2=6.
dtype: Objek yang mendeskripsikan tipe data dari elemen-elemen dalam array. Contohnya adalah int64, float64, atau complex128.
nbytes: Jumlah total byte yang digunakan untuk menyimpan data array di memori. Nilai ini sama dengan size dikalikan dengan ukuran byte per elemen (misalnya, itemsize).
Hubungan antara atribut-atribut ini menjelaskan mengapa NumPy sangat cepat. shape, dtype, dan strides (atribut internal yang terkait dengan nbytes) secara bersama-sama memberi tahu *engine* NumPy cara menghitung lokasi memori pasti dari elemen [i, j, k] menggunakan aritmatika pointer sederhana, tanpa *overhead* dari interpreter Python.
Contoh Kode:
Mari kita buat sebuah array 2D dan periksa atribut-atributnya.

Python


```python
import numpy as np

# Membuat array 2D dari list bersarang
data = np.array([, , ])

print(f"Tipe Objek: {type(data)}")
print(f"Jumlah Dimensi (ndim): {data.ndim}")
print(f"Bentuk (shape): {data.shape}")
print(f"Jumlah Elemen (size): {data.size}")
print(f"Tipe Data (dtype): {data.dtype}")
print(f"Ukuran Memori (nbytes): {data.nbytes} bytes")
```
**Output:**



Tipe Objek: <class 'numpy.ndarray'>
Jumlah Dimensi (ndim): 2
Bentuk (shape): (3, 2)
Jumlah Elemen (size): 6
Tipe Data (dtype): int64
Ukuran Memori (nbytes): 48 bytes
Dari output ini, kita dapat melihat bahwa data adalah array 2D dengan bentuk 3 baris dan 2 kolom. Ia memiliki total 6 elemen, masing-masing adalah integer 64-bit. Ukuran total memori adalah 6 elemen × 8 byte/elemen = 48 byte.

## 2. Tipe Data


### Kontrol Presisi dan Memori

Tidak seperti Python standar yang hanya memiliki tipe int dan float yang umum, NumPy menyediakan berbagai tipe data numerik dengan presisi yang berbeda-beda. Ini termasuk int8, int16, int32, int64 untuk integer, float16, float32, float64 untuk bilangan floating-point, dan complex64, complex128 untuk bilangan kompleks.
Kemampuan untuk memilih tipe data secara spesifik adalah fitur desain yang strategis dalam komputasi ilmiah. Pemilihan dtype adalah sebuah trade-off antara presisi numerik dan penggunaan sumber daya. Misalnya, menggunakan float32 alih-alih float64 (default) akan mengurangi separuh penggunaan memori dan *bandwidth* I/O. Ini sangat signifikan untuk dataset yang sangat besar atau saat melakukan komputasi pada GPU yang memiliki memori terbatas, meskipun dengan risiko kehilangan presisi pada perhitungan yang sensitif.

### Menentukan dan Mengubah Tipe Data

Kita dapat secara eksplisit menentukan tipe data saat membuat array menggunakan argumen dtype. Jika tidak ditentukan, NumPy akan mencoba menebak tipe data yang paling sesuai dari data input.

Python


```python
# Membuat array dengan tipe data yang ditentukan
arr_int = np.array(, dtype=np.int32)
arr_float = np.array(, dtype=np.float64)
arr_complex = np.array(, dtype=np.complex128)

print(f"Array Integer: {arr_int}, dtype: {arr_int.dtype}")
print(f"Array Float: {arr_float}, dtype: {arr_float.dtype}")
print(f"Array Complex: {arr_complex}, dtype: {arr_complex.dtype}")
```
**Output:**



Array Integer: [1 2 3], dtype: int32
Array Float: [1. 2. 3.], dtype: float64
Array Complex: [1.+0.j 2.+0.j 3.+0.j], dtype: complex128
Untuk mengubah tipe data dari array yang sudah ada, kita harus membuat salinan baru menggunakan metode astype(). Metode ini tidak mengubah array asli.

Python


```python
data_float = np.array([1.1, 2.7, 3.5], dtype=np.float64)
print(f"Array Asli: {data_float}, dtype: {data_float.dtype}")

# Typecasting ke integer (bagian desimal akan dipotong/truncated)
data_int = data_float.astype(np.int64)
print(f"Array Hasil Cast: {data_int}, dtype: {data_int.dtype}")
```
**Output:**



Array Asli: [1.1 2.7 3.5], dtype: float64
Array Hasil Cast: [1 2 3], dtype: int64

### Promosi Tipe (Type Promotion)

Ketika operasi aritmatika dilakukan antara array dengan dtype yang berbeda, NumPy secara otomatis akan "mempromosikan" hasilnya ke tipe data yang lebih umum atau presisi untuk mencegah kehilangan informasi.
int32 + int64 → int64
int64 + float32 → float64
float64 + complex64 → complex128

Python


```python
d1 = np.array(, dtype=np.int32)
d2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
hasil = d1 + d2

print(f"Hasil: {hasil}, dtype: {hasil.dtype}") # Hasilnya akan menjadi float64
```
**Output:**



Hasil: [2. 3. 4.], dtype: float64

### Contoh Kasus Kritis: Pentingnya Tipe Data

Pemilihan dtype yang salah dapat menyebabkan hasil yang salah secara matematis. Contoh klasik adalah menghitung akar kuadrat dari bilangan negatif.

Python


```python
# Dengan dtype default (float), akar dari -1 menghasilkan NaN (Not a Number)
arr_real = np.array([-1, 0, 1], dtype=np.float64)
print(f"Akar dengan dtype float: {np.sqrt(arr_real)}")

# Dengan dtype complex, hasilnya benar secara matematis
arr_complex = np.array([-1, 0, 1], dtype=np.complex128)
print(f"Akar dengan dtype complex: {np.sqrt(arr_complex)}")
```
**Output:**



Akar dengan dtype float: [nan  0.  1.]
Akar dengan dtype complex: [0.+1.j 0.+0.j 1.+0.j]
Contoh ini menggarisbawahi bahwa pemahaman dtype sangat penting untuk memastikan kebenaran hasil komputasi.

## 3. Membuat Array dengan Berbagai Teknik

NumPy menyediakan serangkaian fungsi yang komprehensif untuk membuat array, yang masing-masing dirancang untuk alur kerja komputasi ilmiah yang berbeda. Memahami "kapan menggunakan apa" adalah kunci efisiensi.

### Dari Struktur Data Python

Cara paling dasar untuk membuat array adalah dengan mengonversi list atau tuple Python menggunakan np.array(). Ini berguna untuk data kecil atau data yang berasal dari sumber eksternal dalam format list.

Python


```python
# Dari list 1D
vektor = np.array()

# Dari list bersarang (nested) untuk membuat matriks 2D
matriks = np.array([, ])

```
### Array dengan Nilai Konstan

Fungsi-fungsi ini digunakan untuk mengalokasikan memori untuk sebuah array dan mengisinya dengan nilai awal. Ini sering digunakan sebagai "kanvas" yang akan diisi dengan hasil perhitungan nanti.
```python
np.zeros(shape): Membuat array yang semua elemennya bernilai 0.
np.ones(shape): Membuat array yang semua elemennya bernilai 1.
np.full(shape, fill_value): Membuat array yang diisi dengan nilai konstan fill_value.
np.empty(shape): Membuat array tanpa menginisialisasi nilainya. Isinya adalah "sampah" acak dari memori. Ini sedikit lebih cepat daripada zeros karena melewatkan langkah pengisian nilai, dan hanya boleh digunakan jika Anda yakin setiap elemen akan diisi nilainya sebelum dibaca.

```
Python


```python
# Array 2x3 berisi nol
nol = np.zeros((2, 3))

# Array 1D dengan 4 elemen berisi satu
satu = np.ones(4)

# Array 3x3 diisi dengan nilai 5.4
lima_koma_empat = np.full((3, 3), 5.4)

```
### Array dengan Urutan Nilai (Sequences)

Fungsi-fungsi ini sangat penting untuk membuat domain atau sumbu koordinat, misalnya untuk plotting fungsi atau mendefinisikan grid simulasi.
```python
np.arange(start, stop, step): Mirip dengan range Python, tetapi menghasilkan ndarray dan dapat menggunakan step non-integer. Nilai stop tidak termasuk dalam hasilnya.
np.linspace(start, stop, num): Membuat array dengan num elemen, dengan jarak yang sama persis antara start dan stop. Nilai stop secara default termasuk. Ini adalah metode yang lebih disukai untuk membuat sumbu koordinat karena memberikan kontrol presisi atas jumlah titik.
np.logspace(start, stop, num): Mirip linspace, tetapi elemen-elemennya berjarak logaritmik. Menghasilkan num titik antara 10start dan 10stop.

```
Python


```python
# Angka dari 0 hingga 8, dengan langkah 2
a = np.arange(0, 10, 2) # Hasil: [0 2 4 6 8]

# 5 titik dengan jarak sama antara 0 dan 10
b = np.linspace(0, 10, 5) # Hasil: [ 0.   2.5  5.   7.5 10. ]

# 4 titik dengan jarak logaritmik antara 10^0=1 dan 10^2=100
c = np.logspace(0, 2, 4) # Hasil: [  1.           4.64158883  21.5443469  100.        ]

```
## 4. Membuat Matriks

Dalam NumPy, matriks hanyalah sebuah ndarray dua dimensi (ndim=2). Namun, ada beberapa fungsi khusus yang memudahkan pembuatan matriks-matriks umum yang sering digunakan dalam aljabar linear.
```python
np.identity(n): Membuat matriks identitas persegi berukuran n x n, yaitu matriks dengan nilai 1 di diagonal utama dan 0 di tempat lain.
np.eye(N, M=None, k=0): Fungsi yang lebih umum. Membuat matriks N x M dengan nilai 1 di diagonal ke-k. k=0 adalah diagonal utama, k=1 adalah diagonal di atasnya, dan k=-1 adalah diagonal di bawahnya.
np.diag(v, k=0): Fungsi serbaguna. Jika v adalah array 1D, fungsi ini membuat matriks 2D dengan elemen v di sepanjang diagonal ke-k. Jika v adalah matriks 2D, fungsi ini akan mengekstrak diagonal ke-k sebagai array 1D.
```
Contoh Kode: Membuat Matriks Tridiagonal
Matriks tridiagonal, yang memiliki elemen non-nol hanya di diagonal utama, superdiagonal (k=1), dan subdiagonal (k=-1), sangat umum dalam penyelesaian numerik persamaan diferensial.

Python


```python
# Membuat matriks tridiagonal 4x4
# Diagonal utama diisi dengan -2
# Diagonal atas dan bawah diisi dengan 1
matriks_tridiagonal = np.diag(, k=1) + np.diag([-2, -2, -2, -2], k=0) + np.diag(, k=-1)

print(matriks_tridiagonal)
```
**Output:**



[[-2.  1.  0.  0.]
 [ 1. -2.  1.  0.]
 [ 0.  1. -2.  1.]
 [ 0.  0.  1. -2.]]

## 5. Memanipulasi Matriks dan Operasi-Operasi Matriks

Salah satu area yang paling sering menimbulkan kebingungan bagi pemula adalah perbedaan antara operasi *element-wise* dan operasi aljabar linear standar pada matriks.

### Operasi Aritmatika (Element-wise)

Secara default, operator aritmatika standar (+, -, *, /, **) pada array NumPy bekerja secara *element-wise*. Ini berarti operasi tersebut diterapkan pada setiap pasangan elemen yang berada di posisi yang sama di kedua array.

Python


```python
A = np.array([, ])
B = np.array([, ])

# Penjumlahan element-wise
print("A + B =\n", A + B)

# Perkalian element-wise (BUKAN perkalian matriks!)
print("\nA * B =\n", A * B)
```
**Output:**



```python
A + B =
 [[ 6  8]
 [10 12]]

A * B =
 [[ 5 12]
 [21 32]]

```
### Perkalian Matriks (Dot Product)

Untuk melakukan perkalian matriks sesuai aturan aljabar linear, kita harus menggunakan fungsi np.dot() atau, yang lebih modern dan disarankan, operator @.
Keputusan NumPy untuk menggunakan * sebagai perkalian element-wise adalah pilihan desain yang fundamental. Ini mengutamakan konsistensi operasi pada array N-dimensi (di mana perkalian element-wise lebih umum) daripada meniru sintaks dari lingkungan khusus matriks seperti MATLAB. Pengenalan operator @ pada Python 3.5 memberikan sintaks yang bersih untuk kasus aljabar linear tanpa merusak konsistensi *.

Python


```python
# Perkalian matriks yang benar
print("np.dot(A, B) =\n", np.dot(A, B))
print("\nA @ B =\n", A @ B)
```
**Output:**



```python
np.dot(A, B) =
 [[19 22]
 [43 50]]

A @ B =
 [[19 22]
 [43 50]]

```
### Produk Lainnya

NumPy juga menyediakan fungsi untuk jenis produk vektor dan matriks lainnya:
```python
np.inner(a, b): Menghitung *inner product* (produk skalar) dari dua vektor.
np.outer(a, b): Menghitung *outer product* dari dua vektor, yang menghasilkan sebuah matriks.
np.kron(a, b): Menghitung *Kronecker product*, yang merupakan generalisasi dari *outer product* untuk matriks.

```
## 6. Membuat Array Multidimensi

Meskipun matriks (2D) adalah kasus yang paling umum, kekuatan sejati NumPy terletak pada kemampuannya menangani array dengan jumlah dimensi yang lebih tinggi (tensor).

### Generalisasi dari 2D

Array dengan 3 atau lebih dimensi dapat dibuat dengan list bersarang yang lebih dalam.

Python


```python
# Array 3D dengan shape (2, 2, 3)
tensor = np.array([[, ],
                   [, ]])

print(tensor.shape) # Output: (2, 2, 3)

```
### Pembuatan Berbasis Aturan

```python
np.reshape(): Seringkali lebih mudah untuk membuat array 1D yang panjang lalu mengubah bentuknya menjadi multidimensi.
np.meshgrid(): Fungsi ini sangat krusial untuk komputasi pada grid. Diberikan dua vektor koordinat 1D (misalnya, x dan y), meshgrid akan menghasilkan dua matriks koordinat 2D (X dan Y). Matriks X berisi koordinat x untuk setiap titik di grid, dan Y berisi koordinat y. Ini memungkinkan evaluasi fungsi f(x, y) pada seluruh grid secara vektor.
```
**Contoh Kode meshgrid:**

Python


```python
# Vektor koordinat 1D
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)

# Membuat grid koordinat 2D
X, Y = np.meshgrid(x, y)

print("Matriks X:\n", X)
print("\nMatriks Y:\n", Y)

# Mengevaluasi fungsi Z = X^2 + Y^2 pada grid
Z = X**2 + Y**2
print("\nMatriks Z:\n", Z)

```
## 7. Mengetahui Indeks Elemen pada Array

Mengakses dan memanipulasi sebagian data dari array adalah operasi fundamental. NumPy menawarkan sistem pengindeksan yang kaya dan kuat.

### Pengindeksan Dasar dan Slicing

Untuk array 1D, sintaksnya identik dengan list Python: a[index] untuk satu elemen dan a[start:stop:step] untuk *slicing* (membuat irisan). Indeks negatif dihitung dari akhir.
Untuk array multidimensi, kita menggunakan *tuple* indeks yang dipisahkan koma: A[baris, kolom]. Kita dapat menggabungkan indeks integer dan *slice* untuk mengekstrak baris, kolom, atau sub-matriks.

Python


```python
A = np.arange(1, 10).reshape(3, 3)
# A = [,
#      ,
#      ]

# Elemen di baris 1, kolom 2 (indeks 0-based)
print(A) # Output: 6

# Baris pertama (indeks 0)
print(A[0, :]) # Output: [1 2 3]

# Kolom kedua (indeks 1)
print(A[:, 1]) # Output: [2 5 8]

# Sub-matriks 2x2 di pojok kanan atas
print(A[:2, 1:])
# Output:
# [[2 3]
#  [5 6]]

```
### Konsep Kritis: Views vs. Copies

Saat melakukan *slicing* dasar, NumPy tidak membuat salinan data. Sebaliknya, ia membuat sebuah view, yaitu objek ndarray baru yang menunjuk ke data yang sama di memori dengan array asli.
Implikasi: Mengubah nilai dalam *view* akan mengubah array asli. Ini adalah fitur optimisasi yang kuat karena menghindari penyalinan data yang mahal, tetapi bisa menjadi sumber bug yang sulit dilacak jika tidak disadari.

Python


```python
A = np.array()
B_view = A[1:4] # B_view adalah view dari A
print(f"B_view sebelum diubah: {B_view}") # Output: [20 30 40]

B_view = 99 # Mengubah elemen di view
print(f"B_view setelah diubah: {B_view}") # Output: [20 99 40]
print(f"Array A asli juga berubah: {A}") # Output: [10 20 99 40 50]

# Untuk membuat salinan, gunakan.copy()
C_copy = A[1:4].copy()
C_copy = 0
print(f"Array A tidak berubah oleh C_copy: {A}") # Output: [10 20 99 40 50]

```
### Pengindeksan Lanjutan (Fancy & Boolean)

Berbeda dengan *slicing* dasar, pengindeksan lanjutan ini selalu membuat salinan data.
**Fancy Indexing**: Menggunakan list atau array integer untuk memilih elemen-elemen pada indeks tertentu, bahkan jika tidak berurutan.
**Boolean Indexing**: Menggunakan array Boolean dengan shape yang sama sebagai "masker" untuk memfilter elemen. Hanya elemen di mana maskernya True yang akan dipilih. Ini adalah teknik yang sangat kuat untuk analisis data.

Python


```python
data = np.arange(10, 20) # 

# Fancy Indexing: memilih elemen di indeks 0, 3, dan 8
pilihan_fancy = data[]
print(f"Hasil Fancy Indexing: {pilihan_fancy}") # Output: [10 13 18]

# Boolean Indexing: memilih elemen yang lebih besar dari 15
mask = data > 15
pilihan_boolean = data[mask]
print(f"Hasil Boolean Indexing: {pilihan_boolean}") # Output: [16 17 18 19]
```
Ada hierarki efisiensi dalam metode pengindeksan. *Slicing* dasar yang menghasilkan *view* adalah yang tercepat karena hanya memanipulasi metadata. *Fancy* dan *Boolean indexing* lebih lambat karena mereka harus mengalokasikan memori baru dan menyalin data. Memahami perbedaan ini membantu dalam menulis kode yang lebih performan.

## 8. Mengimplementasikan Operasi Logika Matematika Menggunakan Boolean

Array Boolean adalah alat fundamental untuk pemfilteran data dan implementasi logika kondisional secara vektor.

### Membuat dan Menggunakan Array Boolean

Operator perbandingan (>, <, ==, !=) yang diterapkan pada array akan menghasilkan array Boolean baru secara *element-wise*. Array ini kemudian dapat digunakan sebagai masker untuk *Boolean indexing*.

Python


```python
data = np.array([, , ])

# Membuat masker boolean untuk nilai yang lebih besar dari 3
mask = data > 3
print("Masker Boolean:\n", mask)

# Menggunakan masker untuk memfilter data
print("\nElemen > 3:", data[mask]) # Output: [4 5 6]

```
### Logika Kondisional Vektor

Fitur yang sangat kuat dari NumPy adalah bahwa array Boolean diperlakukan sebagai 0 (untuk False) dan 1 (untuk True) dalam operasi aritmatika. Ini memungkinkan kita untuk menulis logika kondisional yang kompleks tanpa menggunakan *loop* atau if statement.

Python


```python
x = np.array([-2, -1, 0, 1, 2])

# Mengatur semua nilai negatif menjadi 0, dan membiarkan yang lain
x_positif = x * (x > 0)
print(f"Hanya nilai positif: {x_positif}") # Output: [0 0 0 1 2]

```
### Fungsi Kondisional np.where

Fungsi np.where(condition, x, y) adalah alternatif vektor untuk ekspresi x if condition else y. Ia mengambil array Boolean condition, dan untuk setiap elemen, ia menghasilkan nilai dari x jika condition adalah True, dan nilai dari y jika False.

Python


```python
# Ganti semua nilai negatif dengan 0, dan nilai non-negatif dengan kuadratnya
hasil = np.where(x < 0, 0, x**2)
print(f"Hasil np.where: {hasil}") # Output: [0 0 0 1 4]

```
## 9. Melakukan Operasi Aritmatika

Operasi aritmatika adalah inti dari komputasi numerik. NumPy membuatnya efisien dan intuitif melalui vektorisasi dan *broadcasting*.

### Operasi Element-wise

Seperti yang telah dibahas, operator aritmatika dasar bekerja elemen per elemen pada array dengan bentuk yang sama.

### Broadcasting: Aturan Fundamental

*Broadcasting* adalah mekanisme yang memungkinkan NumPy melakukan operasi pada array dengan bentuk yang berbeda namun kompatibel. Alih-alih menghasilkan error, NumPy secara virtual "menyiarkan" atau "memperluas" array yang lebih kecil agar cocok dengan bentuk array yang lebih besar.
Aturan *broadcasting* adalah sebagai berikut:
Jika kedua array tidak memiliki jumlah dimensi yang sama, tambahkan dimensi berukuran 1 di sebelah kiri *shape* array yang lebih kecil hingga jumlah dimensinya sama.
Dua array dianggap kompatibel dalam satu dimensi jika ukuran dimensi tersebut sama, atau jika salah satunya berukuran 1.
Penting untuk dipahami bahwa *broadcasting* adalah ilusi yang efisien. NumPy tidak benar-benar membuat salinan data yang diperluas di memori. Sebaliknya, ia secara cerdas menggunakan kembali nilai dari dimensi berukuran 1 dengan memanipulasi *strides* array, sebuah optimisasi tingkat rendah yang membuatnya sangat cepat dan hemat memori.
**Contoh Broadcasting:**

Python


```python
# Matriks 3x3
A = np.ones((3, 3))

# Vektor baris 1x3
b = np.arange(3) # shape (3,) -> diperlakukan sebagai (1, 3)

# Broadcasting: b diperluas secara virtual menjadi matriks 3x3
# [,
#  ,
#  ]
# lalu dijumlahkan dengan A
print("A + b =\n", A + b)

# Vektor kolom 3x1
c = np.arange(3).reshape(3, 1)

# Broadcasting: c diperluas menjadi matriks 3x3
# [,
#  ,
#  ]
print("\nA + c =\n", A + c)

```
## 10. Elementwise Function (ufuncs)

*Universal Functions* atau ufuncs adalah fungsi-fungsi yang beroperasi pada ndarray secara *element-wise*, mendukung *broadcasting*, *type casting*, dan fitur inti NumPy lainnya. Mereka adalah pembungkus (wrapper) yang sangat dioptimalkan di sekitar *loop* C.
**Fungsi Matematika**: np.sqrt, np.exp, np.log, np.abs
**Fungsi Trigonometri**: np.sin, np.cos, np.tan
**Operasi Biner**: np.add, np.multiply, np.maximum (mengembalikan elemen terbesar dari dua array)

Python


```python
x = np.linspace(0, 2*np.pi, 5)
y = np.sin(x)

print(f"x = {x}")
print(f"sin(x) = {y}")

```
## 11. Melakukan Agregasi Beberapa Array

Fungsi agregasi mereduksi sebuah array (atau sebagian dari array) menjadi satu nilai skalar.

### Fungsi Agregasi Umum

Beberapa fungsi agregasi yang paling umum adalah: np.sum, np.mean (rata-rata), np.std (standar deviasi), np.var (varians), np.min, np.max, dan np.prod (produk semua elemen).

### Peran Penting Parameter axis

Secara default, fungsi-fungsi ini bekerja pada seluruh array. Namun, dengan parameter axis, kita dapat mengontrol dimensi di mana agregasi dilakukan.
```python
axis=0: Agregasi dilakukan "ke bawah" sepanjang baris. Hasilnya adalah agregat untuk setiap kolom.
axis=1: Agregasi dilakukan "ke samping" sepanjang kolom. Hasilnya adalah agregat untuk setiap baris.

```
Python


```python
A = np.array([, , ])

# Agregasi global (seluruh elemen)
print(f"Total Sum: {A.sum()}") # Output: 45

# Agregasi sepanjang baris (axis=0) -> menghasilkan jumlah setiap kolom
print(f"Sum of columns: {A.sum(axis=0)}") # Output: [12 15 18]

# Agregasi sepanjang kolom (axis=1) -> menghasilkan jumlah setiap baris
print(f"Sum of rows: {A.sum(axis=1)}") # Output: [ 6 15 24]

```
## 12. Melakukan Operasi dengan Logika Himpunan

NumPy menyediakan fungsi-fungsi untuk melakukan operasi himpunan klasik pada array 1D. Ini sangat berguna untuk membandingkan dan menggabungkan set data.
```python
np.unique(a): Mengembalikan elemen-elemen unik dari array a, diurutkan.
np.intersect1d(a, b): Mengembalikan irisan dari a dan b (elemen yang ada di keduanya).
np.union1d(a, b): Mengembalikan gabungan dari a dan b (semua elemen unik dari keduanya).
np.in1d(a, b): Mengembalikan array Boolean yang menandakan apakah setiap elemen di a juga ada di b.
np.setdiff1d(a, b): Mengembalikan elemen yang ada di a tetapi tidak ada di b.

```
Python


```python
a = np.array()
b = np.array()

print(f"Elemen unik di a: {np.unique(a)}")
print(f"Irisan a dan b: {np.intersect1d(a, b)}")
print(f"Gabungan a dan b: {np.union1d(a, b)}")
print(f"Apakah elemen a ada di b?: {np.in1d(a, b)}")

```
## 13. Memanipulasi Array

Selain operasi matematika, tugas umum lainnya adalah mengubah struktur atau tata letak array. Sebagian besar operasi ini sangat efisien karena hanya memanipulasi metadata (shape, strides) tanpa menyalin data.

### Mengubah Bentuk (Reshaping)

reshape(): Mengubah bentuk array selama jumlah total elemen (size) tetap sama. Menghasilkan *view* jika memungkinkan.
ravel() vs flatten(): Keduanya meratakan array menjadi 1D. ravel() lebih disukai karena menghasilkan *view* jika memungkinkan, sedangkan flatten() selalu membuat salinan data baru.

### Menambah dan Menghapus Dimensi

```python
np.newaxis: Objek khusus yang digunakan dalam *slicing* untuk menambahkan dimensi baru berukuran 1. Sangat berguna untuk mengubah vektor baris menjadi vektor kolom agar sesuai dengan aturan *broadcasting*.
Python
vec = np.array() # shape (3,)
vec_col = vec[:, np.newaxis] # shape (3, 1)
np.expand_dims(): Cara fungsional untuk melakukan hal yang sama seperti newaxis.

```
### Transpose

.T: Atribut yang mudah untuk melakukan transposisi, yaitu membalik urutan sumbu. Untuk matriks 2D, ini menukar baris dan kolom.
```python
np.transpose(): Fungsi yang setara dengan .T.

```
### Menggabungkan dan Memisahkan Array

Operasi-operasi ini, tidak seperti *reshaping*, melibatkan penyalinan data dan karenanya lebih lambat.
```python
np.concatenate((a, b), axis=...): Fungsi paling umum untuk menggabungkan array di sepanjang sumbu yang ditentukan.
np.vstack((a, b)): Pintasan untuk np.concatenate di sepanjang axis=0 (menumpuk secara vertikal).
np.hstack((a, b)): Pintasan untuk np.concatenate di sepanjang axis=1 (menumpuk secara horizontal).

```
## Kesimpulan

Penguasaan NumPy adalah langkah fundamental dan tak terhindarkan bagi siapa pun yang serius dalam bidang komputasi ilmiah, rekayasa, dan ilmu data menggunakan Python.
Kekuatan sejati NumPy terletak pada tiga pilar utama:
**Struktur Data yang Efisien**: ndarray menyediakan penyimpanan data numerik yang padat dan cepat diakses.
**Vektorisasi**: Kemampuan untuk melakukan operasi batch pada seluruh array menggantikan *loop* Python yang lambat, menghasilkan peningkatan kinerja yang dramatis.
**Broadcasting**: Mekanisme cerdas dan hemat memori yang memungkinkan operasi intuitif pada array dengan bentuk yang berbeda.
Dengan memahami konsep-konsep ini termasuk atribut array, tipe data, berbagai metode pembuatan, perbedaan antara operasi *element-wise* dan aljabar linear, serta teknik pengindeksan dan manipulasi. Pengetahuan ini tidak hanya memungkinkan untuk menulis kode numerik yang lebih cepat dan bersih, tetapi juga merupakan prasyarat esensial untuk melangkah ke pustaka tingkat lebih tinggi dalam ekosistem Python, seperti SciPy untuk algoritma ilmiah yang lebih kompleks, Pandas untuk analisis data tabular, dan Scikit-learn untuk aplikasi *machine learning*.
