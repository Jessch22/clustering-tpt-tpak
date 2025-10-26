judul = "Sistem Pemetaan Wilayah Berdasarkan Tingkat Pengangguran Terbuka dan Tingkat Partisipasi Angkatan Kerja di Indonesia dengan K-Means dan DBSCAN"
string1 = """ Ketenagakerjaan merupakan aspek penting dalam pembangunan nasional karena berpengaruh langsung terhadap kesejahteraan masyarakat. 
Berdasarkan data BPS, Tingkat Pengangguran Terbuka (TPT) menurun pada 2024 menjadi 4,91% dari 5,32% pada 2023, 
sedangkan Tingkat Partisipasi Angkatan Kerja (TPAK) meningkat menjadi 70,63% dari 69,48%.

TPT dan TPAK menjadi dua indikator penting dalam ketenagakerjaan:
    <ul>
    <li>Tingkat Pengangguran Terbuka (TPT) = persentase penduduk angkatan kerja yang belum bekerja/pengangguran, tidak termasuk yang tidak mencari kerja.</li>
    <li>Tingkat Partisipasi Angkatan Kerja (TPAK) = proporsi angkatan kerja terhadap penduduk usia kerja, yang secara aktif mencari pekerjaan atau sudah bekerja.</li>
    </ul>
"""

string2 = """Meskipun Tingkat Pengangguran Terbuka (TPT) nasional menurun dan Tingkat Partisipasi Angkatan Kerja (TPAK) meningkat, ketimpangan masih terjadi antarwilayah.
Oleh karena itu, diperlukan analisis clustering wilayah berdasarkan kedua indikator tersebut untuk memahami pola ketenagakerjaan di Indonesia.
Kombinasi keduanya dapat menggambarkan dinamika pasar kerja:
<ul>
<li>TPT & TPAK tinggi → banyak pencari kerja belum terserap.
<li>TPT & TPAK rendah → potensi tenaga kerja belum dimanfaatkan optimal.
</ul>
"""

string3 = """
Untuk mengenali pola dan kelompok wilayah dengan karakteristik serupa, 
penelitian ini membuat sistem pemetaan berbasis web menggunakan Streamlit.
Sistem ini memanfaatkan algoritma K-Means dan DBSCAN untuk melakukan clustering wilayah berdasarkan TPT dan TPAK.
Sistem menampilkan hasil berupa peta interaktif, box plot, dan evaluasi otomatis dengan Silhouette Score dan Davies-Bouldin Index. 
Tujuannya agar analisis ketenagakerjaan dapat dilakukan secara visual, efisien, dan mudah dipahami oleh publik maupun instansi terkait.
"""

cara_penggunaan = """
<div style="background-color:#f0f0f0; padding:15px; border-radius:10px;">
    <ol>
        <li>Buka <b>Halaman Clustering</b> untuk mulai analisis.</li>
        <li>Pilih variabel (<b>TPT</b>/<b>TPAK</b>) dan rentang tahun.</li>
        <li>Tentukan metode dan parameter yang diinginkan.</li>
        <li>Klik <b>'Jalankan Clustering'</b>.</li>
        <li>Lihat hasil pada halaman Clustering.</li>
        <li>Akses contoh dataset melalui <b>Halaman Dataset</b>.</li>
    </ol>
</div>
"""

question1 = """Apa itu Tingkat Pengangguran Terbuka (TPT) dan Tingkat Partisipasi Angkatan Kerja (TPAK)?"""
question2 = """Metode clustering dan metrik evaluasi apa yang digunakan dalam sistem ini?"""

answer1 = """
<div style="background-color:#f0f0f0; padding:15px; border-radius:10px;">
    <p><b>Tingkat Pengangguran Terbuka (TPT)</b> dan <b>Tingkat Partisipasi Angkatan Kerja (TPAK)</b> merupakan indikator utama ketenagakerjaan di Indonesia.</p>
    <p>Dengan memetakan TPT dan TPAK, sistem ini membantu memahami kondisi pasar kerja dan ketimpangan antarwilayah:</p>
    <ul>
        <li><b>TPT</b> menunjukkan persentase penduduk yang aktif mencari kerja namun belum bekerja — mencerminkan penyerapan tenaga kerja.</li>
        <li><b>TPAK</b> mengukur proporsi penduduk usia produktif yang aktif secara ekonomi, baik bekerja maupun mencari kerja.</li>
    </ul>
</div><br>
"""

answer2 = """
<div style="background-color:#f0f0f0; padding:15px; border-radius:10px;">
    <p>Sistem ini menggunakan dua algoritma clustering:</p>
    <h5>1. K-Means</h5>
    <ul>
        <li>Mengelompokkan data menjadi <i>k</i> klaster berdasarkan rata-rata terdekat.</li>
        <li>Cocok untuk data yang terdistribusi normal dan seimbang.</li>
        <li>Dapat dipilih jika ingin mengelompokkan data menjadi <i>k</i> klaster.</li>
    </ul>
    <h5>2. DBSCAN</h5>
    <ul>
        <li>Membentuk klaster berdasarkan kepadatan titik (<i>density-based</i>).</li>
        <li>Cocok untuk data tidak beraturan atau tidak terdistribusi normal.</li>
        <li>Dapat dipilih jika ingin menemukan klaster dengan bentuk bebas dan mengidentifikasi <i>outlier</i>.</li>
    </ul>
    <br>
    <h5>Metrik Evaluasi:</h5>
    <ul>
        <li><b>Silhouette Score:</b> menilai seberapa baik data terbagi ke dalam cluster. Semakin tinggi nilai Silhouette, data dalam cluster semakin sesuai dengan tempatnya</li>
        <li><b>Davies–Bouldin Index (DBI):</b> menilai kualitas pemisahan cluster. Semakin nilai <b>lebih kecil</b> = kualitas cluster lebih baik dan bersifat kompak</li>
    </ul>
</div><br>
"""

penjelasan_dataset = """
<div style="background-color:#f0f0f0; padding:15px; border-radius:10px;">
<ul>
    <li><b>Provinsi</b>: Nama provinsi.</li>
    <li><b>Kabupaten/Kota</b>: Nama kabupaten/kota.</li>
    <li><b>TPT_2018</b>: Tingkat pengangguran terbuka tahun 2018.</li>
    <li><b>TPT_2019</b>: Tingkat pengangguran terbuka tahun 2019.</li>
    <li><b>TPT_2020</b>: Tingkat pengangguran terbuka tahun 2020.</li>
    <li><b>TPT_2021</b>: Tingkat pengangguran terbuka tahun 2021.</li>
    <li><b>TPT_2022</b>: Tingkat pengangguran terbuka tahun 2022.</li>
    <li><b>TPT_2023</b>: Tingkat pengangguran terbuka tahun 2023.</li>
    <li><b>TPT_2024</b>: Tingkat pengangguran terbuka tahun 2024.</li>
    <li><b>TPAK_2018</b>: Tingkat partisipasi angkatan kerja tahun 2018.</li>
    <li><b>TPAK_2019</b>: Tingkat partisipasi angkatan kerja tahun 2019.</li>
    <li><b>TPAK_2020</b>: Tingkat partisipasi angkatan kerja tahun 2020.</li>
    <li><b>TPAK_2021</b>: Tingkat partisipasi angkatan kerja tahun 2021.</li>
    <li><b>TPAK_2022</b>: Tingkat partisipasi angkatan kerja tahun 2022.</li>
    <li><b>TPAK_2023</b>: Tingkat partisipasi angkatan kerja tahun 2023.</li>
    <li><b>TPAK_2024</b>: Tingkat partisipasi angkatan kerja tahun 2024.</li>
    </ul>
</div>"""

# PENJELASAN
penjelasan_tpt = "Tingkat Pengangguran Terbuka (TPT) adalah persentase jumlah pengangguran terhadap angkatan kerja."
penjelasan_tpak = "Tingkat Partisipasi Angkatan Kerja (TPAK) adalah persentase jumlah angkatan kerja terhadap total populasi usia kerja."
penjelasan_kmeans = "Pilih K-Means jika ingin mengelompokkan wilayah berdasarkan jumlah cluster yang telah ditentukan."
penjelasan_dbscan = "Pilih DBSCAN jika ingin menemukan cluster dengan bentuk bebas dan mengidentifikasi outlier."