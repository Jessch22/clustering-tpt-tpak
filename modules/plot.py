import streamlit as st
import io
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.metrics import silhouette_samples
import numpy as np
import folium
import matplotlib.colors as mcolors
import plotly.figure_factory as ff
import plotly.graph_objects as go

def get_cluster_color_map(cluster_ids):
    """Mendapatkan peta warna untuk cluster berdasarkan cluster_ids.

    - cluster_ids: iterable berisi id cluster (bisa int atau str). Noise harus diberi id -1.
    - Warna abu-abu (#D3D3D3) DILARANG untuk cluster valid; hanya dipakai untuk noise (-1).
    - Fungsi mengembalikan dict: {cluster_id: hexcolor, ..., -1: '#D3D3D3'}.
    """
    color_map = {}

    # Filter cluster_ids valid (bukan NaN atau -1)
    valid_cluster_ids = sorted([c for c in cluster_ids if pd.notna(c) and c != -1], key=lambda x: (int(x) if str(x).lstrip('-').isdigit() else str(x)))
    n_colors = len(valid_cluster_ids)

    # Ambil colormap 'tab10' dari Matplotlib sebagai preferensi
    try:
        cmap = plt.colormaps.get('tab10')
        if n_colors > 0:
            for i, cluster_id in enumerate(valid_cluster_ids):
                # normalisasi indeks warna ke [0,1]
                norm_index = float(i) / (n_colors - 1) if n_colors > 1 else 0.5
                rgba_color = cmap(norm_index)
                hex_color = mcolors.to_hex(rgba_color)
                # Pastikan warna tidak sama dengan gray noise '#D3D3D3'
                if hex_color.lower() in ('#d3d3d3', '#d3d3d'):  # defensif check
                    # fallback ke warna lain dari plotly
                    hex_color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                color_map[cluster_id] = hex_color
    except Exception as e:
        st.warning(f"Gagal mendapatkan colormap Matplotlib ('tab10') ({e}), fallback ke Plotly.")
        plotly_colors = px.colors.qualitative.Plotly
        for i, cluster_id in enumerate(valid_cluster_ids):
            color_map[cluster_id] = plotly_colors[i % len(plotly_colors)]

    # Pastikan semua cluster valid memiliki warna (fallback bila ada edge cases)
    if valid_cluster_ids:
        fallback_palette = px.colors.qualitative.Plotly
        for i, cluster_id in enumerate(valid_cluster_ids):
            if cluster_id not in color_map or not color_map[cluster_id]:
                color_map[cluster_id] = fallback_palette[i % len(fallback_palette)]

    # Warna khusus untuk noise (-1) — harus abu-abu
    color_map[-1] = '#D3D3D3'

    return color_map

def render_metrics_and_silhouette(scores, hasil_data, data_for_clustering):
    """Render metrik evaluasi dan silhouette plot"""
    
    # Jika ada skor, tampilkan metriknya
    # Membuat 3 kolom untuk metrik: Silhouette, DBI, Waktu
    if scores:
        met_col1, met_col2, met_col3 = st.columns(3)
        sil_score = scores.get('silhouette', None)
        dbi_score = scores.get('dbi', None)
        time_sec = scores.get('time_sec', None)
        
        help_sil = """
        Nilai silhouette (SC: Silhouette Coefficient) menunjukkan keberadaan struktur di cluster.
        - SC ≥ 0.70: Struktur sangat kuat
        - 0.50 ≤ SC < 0.70: Struktur menengah
        - 0.25 ≤ SC < 0.50: Struktur lemah
        - SC < 0.25: Tidak ada struktur yang jelas
        """
        help_dbi = "Semakin kecil nilai DBI, cluster lebih kompak dan terpisah, menandakan sedikit atau tidak ada tumpang tindih."
        help_time = "Menunjukkan lama waktu eksekusi algoritma clustering."
        
        met_col1.metric("Silhouette", f"{sil_score:.4f}" if sil_score is not None else "N/A", help=help_sil)
        met_col2.metric("DBI", f"{dbi_score:.4f}" if dbi_score is not None else "N/A", help=help_dbi)
        met_col3.metric("Waktu", f"{time_sec:.2f} s" if time_sec is not None else "N/A", help=help_time)
        
    else:
        # Placeholder jika tidak ada skor
        st.container(height=80)
        
    # ============================================================
    st.divider()
    # ============================================================
    
    # Panggil fungsi render_silhouette_plot (Silhouette Plot)
    render_silhouette_plot(data_for_clustering, hasil_data, scores)

def create_folium_map(gdf_merged, key_column='WADMKK', tooltip_name_col: str = 'display_name', tooltip_prov_col: str = 'prov'):
    """
    Membuat peta interaktif Folium berdasarkan GeoDataFrame, dengan tooltip dan legenda
    """
    
    # Cek validitas data:
    # Jika gdf_merged tidak valid, kembalikan None
    if gdf_merged is None or gdf_merged.empty or 'geometry' not in gdf_merged.columns:
        st.warning("Data geospasial tidak valid untuk membuat peta."); return None
    if 'Cluster' not in gdf_merged.columns:
        st.warning("Kolom 'Cluster' tidak ditemukan di data geospasial."); return None

    # Pastikan kolom tooltip ada; jika tidak, buat fallback yang readable
    if tooltip_name_col not in gdf_merged.columns:
        st.warning(f"Kolom tooltip nama ('{tooltip_name_col}') tidak ditemukan. Menggunakan kolom '{key_column}' sebagai pengganti.")
        gdf_merged[tooltip_name_col] = gdf_merged.get(key_column, "").astype(str)
    if tooltip_prov_col not in gdf_merged.columns:
        gdf_merged[tooltip_prov_col] = gdf_merged.get('prov', "").astype(str)

    try:
        # Koordinat pusat Indonesia
        map_center = [-2.5489, 118.0149]; zoom_level = 5
        m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="cartodbpositron")

        # Menentukan warna cluster
        unique_clusters_raw = gdf_merged['Cluster'].unique()
        clusters_valid = sorted([c for c in unique_clusters_raw if pd.notna(c) and c != -1],
                                key=lambda x: (int(x) if str(x).lstrip('-').isdigit() else str(x)))
        color_dict = get_cluster_color_map(unique_clusters_raw)

        # Fungsi untuk mendapatkan warna berdasarkan cluster
        def _get_color_for_feature(cluster_id):
            # cluster_id bisa berupa int, float, atau string; periksa beberapa bentuk kunci
            try:
                # Jika null/NaN -> treat as NA -> use noise color
                if cluster_id is None or (isinstance(cluster_id, float) and np.isnan(cluster_id)):
                    return color_dict.get(-1, '#D3D3D3')
                # Coba lookup langsung
                if cluster_id in color_dict:
                    return color_dict[cluster_id]
                # Coba convert ke int jika memungkinkan
                try:
                    cid_int = int(cluster_id)
                    if cid_int in color_dict:
                        return color_dict[cid_int]
                except Exception:
                    pass
                # Coba string form
                cid_str = str(cluster_id)
                if cid_str in color_dict:
                    return color_dict[cid_str]
            except Exception:
                pass
            # Fallback: jangan pakai abu-abu untuk cluster; gunakan hitam atau warna primer dari Plotly
            return '#000000'

        # Tooltip fields and aliases
        fields_for_tooltip = [tooltip_prov_col, tooltip_name_col, 'Cluster']
        aliases_for_tooltip = ['Provinsi:', 'Wilayah:', 'Cluster:']

        geojson = folium.GeoJson(
            gdf_merged,
            style_function=lambda feature: {
                'fillColor': _get_color_for_feature(feature['properties'].get('Cluster')),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            highlight_function=lambda x: {'weight': 2, 'color': 'red'},
            tooltip=folium.features.GeoJsonTooltip(
                fields=fields_for_tooltip, aliases=aliases_for_tooltip,
                localize=True, sticky=False, labels=True,
                style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; box-shadow: 3px;",
                max_width=800,
            ),
            popup=folium.features.GeoJsonPopup(
                fields=fields_for_tooltip, aliases=aliases_for_tooltip, localize=True
            )
        )
        geojson.add_to(m)

        # Buat legend kustom: hanya buat entry untuk cluster valid + noise
        if clusters_valid:
            legend_html = '''
                <div style="position:fixed; bottom:50px; right:50px; width:170px; height:auto; 
                border:2px solid grey; z-index:9999; font-size:12px; background-color:white; 
                padding:10px; opacity:0.95;"><b>Legenda Cluster</b><br>
            '''
            for cluster_id in clusters_valid:
                col = color_dict.get(cluster_id, '#000000')
                legend_html += f'&nbsp; <i style="background:{col}; width:15px; height:15px; display:inline-block; margin-right:5px; border: 1px solid grey;"></i> Cluster {cluster_id}<br>'
            # Noise entry
            has_noise = (-1 in unique_clusters_raw) or (gdf_merged['Cluster'].isnull().any())
            if has_noise:
                color_na = color_dict.get(-1, '#D3D3D3')
                legend_html += f'&nbsp; <i style="background:{color_na}; width:15px; height:15px; display:inline-block; margin-right:5px; border: 1px solid grey;"></i> Noise / N/A<br>'
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))

        return m
    except Exception as e:
        st.error(f"Gagal membuat peta Folium: {e}")
        return None

def render_boxplot(df_hasil, data_for_clustering=None):
    """Membuat dan menampilkan boxplot interaktif menggunakan pd.melt."""
    # Jika df_hasil tidak valid, tampilkan info dan keluar
    if df_hasil is None or df_hasil.empty: st.info("Belum menjalankan clustering."); return
    # Jika kolom 'Cluster' tidak ada, tampilkan peringatan dan keluar
    if 'Cluster' not in df_hasil.columns: st.warning("Kolom 'Cluster' tidak ditemukan."); return
    # Jika tidak ada cluster valid, tampilkan info dan keluar
    if df_hasil['Cluster'].nunique() < 1 or (df_hasil['Cluster'].nunique() == 1 and df_hasil['Cluster'].unique()[0] == -1): st.info("Tidak ada cluster valid."); return

    try:
        # Jika data_for_clustering disediakan, gunakan kolomnya sebagai value_vars untuk dimelt
        if data_for_clustering is not None and not data_for_clustering.empty:
            value_vars = data_for_clustering.columns.tolist()
        else:
            # Jika tidak, gunakan semua kolom numerik kecuali 'Cluster' untuk dimelt
            potential_value_vars = df_hasil.select_dtypes(include=np.number).columns.tolist()
            exclude_cols = ['Cluster']
            value_vars = [col for col in potential_value_vars if col not in exclude_cols and '_' in col]
            if not value_vars: st.error("Tidak ada kolom yang valid untuk ditampilkan."); return
            
        # Persiapan data untuk boxplot
        # Mengambil kolom kab_kota dan Cluster sebagai id_vars agar tetap ada di data setelah melt
        id_vars = ['kab_kota', 'Cluster']
        
        df_hasil_melt = df_hasil.copy()
        # Kolom kab_kota dijadikan string untuk menghindari masalah tipe data
        df_hasil_melt['kab_kota'] = df_hasil_melt['kab_kota'].astype(str)
        # Melting data dari wide ke long format
        df_long = pd.melt(df_hasil_melt, id_vars=id_vars, value_vars=value_vars, var_name='Variable_Tahun', value_name='Nilai')
        
        try:
            # Memisahkan nama variabel dan tahun dari kolom 'Variable_Tahun'
            split_data = df_long['Variable_Tahun'].str.extract(r'^(.*?)_(\d{4})$')
            # Jika split_data masih kosong, menggunakan metode lain untuk memisahkan
            if split_data.isnull().all().all(): split_data = df_long['Variable_Tahun'].str.rsplit('_', n=1, expand=True)
            # Assign ke kolom baru: Variabel dan Tahun
            df_long['Variabel'] = split_data[0]
            df_long['Tahun'] = split_data[1]
            
            # Hapus baris dengan nilai NaN pada kolom Variabel atau Tahun
            df_long.dropna(subset=['Variabel', 'Tahun'], inplace=True)
            
            # Jika df_long masih kosong, tampilkan info dan keluar
            if df_long.empty: st.error("Tidak ada data untuk ditampilkan."); return
            
            # Mengurutkan tahun secara kronologis
            unique_years = sorted(df_long['Tahun'].unique())
            df_long['Tahun'] = pd.Categorical(df_long['Tahun'], categories=unique_years, ordered=True)
        
        except Exception as e_split:
            st.error(f"Gagal saat memisahkan variabel dan tahun. Error: {e_split}")
            return
        
        # Mengurutkan cluster valid untuk konsistensi warna
        unique_clusters_valid = sorted([c for c in df_long['Cluster'].unique() if pd.notna(c) and c != -1])
        # Kolom Cluster dijadikan string untuk keperluan pewarnaan di Plotly
        df_long['Cluster_Str'] = df_long['Cluster'].astype(str)
        # Mengurutkan kategori cluster sebagai string
        cluster_categories_str = sorted([str(c) for c in unique_clusters_valid], key=lambda x: int(x))
        # Jika ada noise (-1), tambahkan ke urutan kategori
        if '-1' in df_long['Cluster_Str'].unique(): cluster_categories_str.append('-1')
        
        # Memanggil fungsi get_cluster_color_map
        color_map_plotly = get_cluster_color_map(unique_clusters_valid)
        # Mengubah key ke string untuk dicocokkan dengan Cluster_Str
        color_discrete_map_box = {str(k): v for k, v in color_map_plotly.items()}
        # Jika ada noise (-1), tambahkan ke peta warna
        if '-1' in df_long['Cluster_Str'].unique():
            color_discrete_map_box['-1'] = color_map_plotly.get(-1, 'lightgrey')
            
        # Membuat boxplot menggunakan Plotly Express
        fig_box = px.box(
            # data
            df_long, x='Tahun', y='Nilai',
            # warna berdasarkan cluster
            color='Cluster_Str',
            # pisah berdasarkan variabel
            facet_col='Variabel',
            # judul
            title="Distribusi Nilai Variabel per Cluster dan Tahun",
            # urutan kategori berdasarkan tahun dan cluster
            category_orders={
                "Tahun": unique_years,
                "Cluster_Str": cluster_categories_str
            },
            color_discrete_map=color_discrete_map_box,
            facet_col_wrap=4, facet_col_spacing=0.03, facet_row_spacing=0.07
        )
        
        # Update judul x dan y axis
        fig_box.update_yaxes(matches=None, showticklabels=True, title_text="")
        fig_box.update_xaxes(title_text="Tahun")
        
        # Update ukuran layout dan judul legenda
        fig_box.update_layout(height=500, boxmode='group', legend_title_text='Cluster')
        fig_box.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_box, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal total saat membuat box plot: {e}")

def render_silhouette_plot(data_for_clustering, hasil_data, scores):
    """Membuat silhouette plot Matplotlib dengan warna konsisten."""
    # Jika data tidak valid, tampilkan info dan keluar
    if data_for_clustering is None or hasil_data is None or scores is None: st.info("Belum menjalankan clustering."); return
    
    # Mengambil label unik
    labels = hasil_data['Cluster']
    unique_labels = labels.unique()
    
    # Jumlah cluster valid berdasarkan label unik
    n_clusters_valid = len([l for l in unique_labels if l != -1 and pd.notna(l)])
    
    # Jika kurang dari 2 cluster valid, tampilkan peringatan dan keluar
    if n_clusters_valid < 2: 
        st.warning("Silhouette plot memerlukan minimal 2 cluster valid.")
        with st.expander("Mengapa Silhouette Plot tidak muncul?"):
            st.markdown("""
                Silhouette plot berguna untuk membandingkan seberapa mirip sebuah titik dengan **cluster-nya sendiri** dibandingkan dengan **cluster tetangga terdekat**.
                Agar perbandingan ini bisa dilakukan, sistem membutuhkan minimal **dua cluster valid** (selain Noise).
                #### 1. Semua Wilayah Menjadi "Noise" (Label -1)
                - Arti: Tidak ada wilayah yang cukup padat untuk membentuk cluster.
                - Penyebab: Nilai **Epsilon** terlalu kecil atau **MinPts** terlalu tinggi.

                #### 2. Semua Wilayah Menjadi 1 Cluster (Label 0)
                - Arti: Semua wilayah termasuk dalam satu cluster besar tanpa ada yang dianggap Noise.
                - Penyebab: Nilai **Epsilon** terlalu besar. Semua titik dianggap 'tetangga' satu sama lain.

                #### 3. Hanya Ada 1 Cluster + Noise (Label 0 dan -1)
                - Arti: Hanya ada satu cluster yang terbentuk, sementara wilayah lainnya dianggap Noise karena terlalu tersebar.
                - Penyebab: Nilai **Epsilon** dan **MinPts** sudah tepat untuk membentuk satu cluster, tetapi tidak cukup untuk membentuk cluster tambahan.
                """)
        return
    
    # Mengambil nilai silhouette rata-rata. Jika tidak ada, tampilkan peringatan dan keluar
    silhouette_avg = scores.get('silhouette')
    if silhouette_avg is None: st.warning("Silhouette score tidak ditemukan."); return
    
    
    try:
        sample_silhouette_values = silhouette_samples(data_for_clustering, labels)
        st.caption("Bar yang mengarah ke kiri menunjukkan nilai silhouette negatif, dan lebar/tinggi tiap blok mewakili ukuran cluster sesuai jumlah anggotanya.")
    except Exception as e: 
        st.error(f"Gagal membuat sample silhouette values: {e}"); return
        
    # Membuat plot silhouette dengan ketentuan size: 7x5 dan posisi awal y_lower = 10
    fig, ax = plt.subplots(figsize=(7, 5)); y_lower = 10
    # Mengurutkan cluster valid untuk konsistensi warna
    cluster_labels_sorted = sorted([label for label in unique_labels if label != -1 and pd.notna(label)])
    # Mengambil warna konsisten
    color_map_sil = get_cluster_color_map(cluster_labels_sorted)
    
    # Untuk setiap cluster valid, plot silhouette values
    for i in cluster_labels_sorted:
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]; ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]; y_upper = y_lower + size_cluster_i
        color = color_map_sil.get(i, 'black')
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)); y_lower = y_upper + 10
        
    # Set title dan label
    ax.set_title("Silhouette Plot"); ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Cluster")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--"); ax.set_yticks([])
    ax.set_xticks(np.arange(-0.1, 1.1, 0.2)); ax.set_xlim([-0.1, 1.0])

    try: # Simpan & tampilkan
        img_buffer = io.BytesIO(); fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150); img_buffer.seek(0)
        st.image(img_buffer, use_container_width=True); plt.close(fig)
    except Exception as e: st.error(f"Error saat menyimpan gambar: {e}"); plt.close(fig)

def render_scatter_plots(df_hasil, data_for_clustering=None):    
    # Jika df_hasil tidak valid, tampilkan info dan keluar
    if df_hasil is None or df_hasil.empty: 
        st.info("Belum menjalankan clustering.")
        return
    # Jika kolom 'Cluster' tidak ada, tampilkan peringatan dan keluar
    if 'Cluster' not in df_hasil.columns: 
        st.warning("Kolom 'Cluster' tidak ditemukan.")
        return
    # Jika data_for_clustering tidak valid, tampilkan info dan keluar
    if data_for_clustering is None or data_for_clustering.empty: 
        st.info("Data untuk clustering tidak valid.")
        return
    
    # Mendapat variabel untuk ditampilkan
    scatter_vars = data_for_clustering.columns.tolist()
    D = len(scatter_vars)
    
    if D < 2: 
        st.info("Pair Plot memerlukan minimal 2 dimensi untuk perbandingan.")
        return
    
    try:
        # Persiapan data untuk plot
        # Ambil semua kolom variabel yang digunakan
        cols_for_plot_data = [col for col in scatter_vars if col in df_hasil.columns]
        # Tambahkan kolom esensial untuk hover, pastikan ada
        for c in ['Cluster', 'kab_kota', 'prov']:
            if c in df_hasil.columns and c not in cols_for_plot_data:
                cols_for_plot_data.append(c)
                
        plot_data = df_hasil[cols_for_plot_data].copy()
        plot_data.dropna(subset=scatter_vars, inplace=True)
        
        if plot_data.empty: 
            st.warning("Tidak ada data untuk ditampilkan setelah menghapus missing values.")
            return
        
        # Ubah cluster ke string agar Plotly memperlakukannya sebagai kategori
        plot_data['Cluster_Str'] = plot_data['Cluster'].astype(str)
        
        # 1. Dapatkan semua label cluster (string) unik
        all_cluster_labels_str = plot_data['Cluster_Str'].unique()
        
        # 2. Sortir label tersebut secara numerik (agar -1, 0, 1, 2...)
        cluster_list_sorted_str = sorted(
            all_cluster_labels_str, 
            key=lambda x: int(x) if x.lstrip('-').isdigit() else x
        )
        
        # 3. Buat dictionary untuk 'category_orders' yang akan digunakan px.scatter
        category_orders_dict = {"Cluster_Str": cluster_list_sorted_str}
        
        # 4. Dapatkan peta warna (sudah diurutkan dengan benar oleh get_cluster_color_map)
        cluster_categories_numeric = sorted([c for c in plot_data['Cluster'].unique() if c != -1 and pd.notna(c)])
        color_map = get_cluster_color_map(cluster_categories_numeric) #
        color_map_str = {str(k): v for k, v in color_map.items()}
        
        # Pastikan noise (-1) juga memiliki warna jika ada
        if '-1' in cluster_list_sorted_str:
            color_map_str['-1'] = color_map.get(-1, 'lightgrey')
        
        # === 3. Fokus ke Variabel Tertentu ===
        
        focal_var = st.selectbox(
            "Pilih Variabel Fokus (untuk Sumbu Y)",
            scatter_vars,
            index=0,
            key="scatter_focal",
            help="Pilih satu variabel untuk dilihat perbandingannya dengan semua variabel lain."
        )
        if not focal_var:
            return

        # 4. Plot Diagonal (Distribusi) untuk Variabel Fokus
        hist_data = []
        group_labels = []
        cluster_list_sorted_str = sorted(
            plot_data['Cluster_Str'].unique(), 
            key=lambda x: int(x) if x.lstrip('-').isdigit() else x
        )
        
        for cluster_str in cluster_list_sorted_str:
            data_for_cluster = plot_data[plot_data['Cluster_Str'] == cluster_str][focal_var].dropna()
            if not data_for_cluster.empty:
                hist_data.append(data_for_cluster)
                group_labels.append(f"Cluster {cluster_str}")

        if not hist_data:
            st.warning(f"Tidak ada data valid untuk plot diagonal {focal_var}")
        else:
            colors_for_plot = [color_map_str.get(c_str, 'black') for c_str in cluster_list_sorted_str]

            # PLOT KDE
            try:
                fig_diag = ff.create_distplot(
                    hist_data, 
                    group_labels, 
                    colors=colors_for_plot,
                    show_hist=False,  # Sembunyikan histogram balok
                    show_rug=False    # Sembunyikan rug plot
                )
                
                for trace in fig_diag.data:
                    trace.update(fill='tozeroy', opacity=0.5)
                    
                fig_diag.data = fig_diag.data[::-1]

                fig_diag.update_layout(
                    title=f"Distribusi (KDE) untuk {focal_var}", 
                    legend_title_text='Cluster',
                    xaxis_title=focal_var,
                    yaxis_title="Kepadatan (Density)",
                )
                st.plotly_chart(fig_diag, use_container_width=True)
                
            except Exception as e_kde:
                st.error(f"Gagal membuat plot KDE: {e_kde}. Fallback ke histogram grup.")
                fig_diag_fallback = px.histogram(
                    plot_data, 
                    x=focal_var,
                    color="Cluster_Str",
                    color_discrete_map=color_map_str, 
                    barmode='group',
                    category_orders=category_orders_dict
                )
                st.plotly_chart(fig_diag_fallback, use_container_width=True)

        # 5. Plot Scatter "Satu-vs-Semua"
        st.markdown(f"**Hubungan (Scatter Plot) antara {focal_var} dan Variabel Lain**")
        
        # Dapatkan semua variabel LAINNYA untuk sumbu X
        other_vars = [v for v in scatter_vars if v != focal_var]
        
        if not other_vars:
            st.info("Hanya ada 1 dimensi, tidak ada variabel lain untuk dibandingkan.")
            return

        # Atur layout kolom (misal: 3 plot per baris)
        cols = st.columns(3) 
        
        for i, other_var in enumerate(other_vars):
            with cols[i % 3]: # Ini akan memutar (0, 1, 2, 0, 1, 2, ...)
                fig_single = px.scatter(
                    plot_data,
                    x=other_var,    # Variabel lain di Sumbu X
                    y=focal_var,    # Variabel Fokus di Sumbu Y
                    color="Cluster_Str",
                    color_discrete_map=color_map_str,
                    hover_data=['kab_kota', 'prov', 'Cluster'], # Hover interaktif
                    title=f"{focal_var} vs {other_var}",
                    category_orders=category_orders_dict
                )
                # Perkecil judul agar muat
                fig_single.update_layout(
                    title_font_size=12, 
                    height=350,
                )
                st.plotly_chart(fig_single, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membuat pair plot: {e}")
        st.exception(e)

# ============================================================
# FUNGSI UNTUK RENDER KE BUFFER (UNTUK PDF)
# ============================================================

def render_static_map_to_buffer(gdf_to_plot):
    if gdf_to_plot is None or gdf_to_plot.empty or 'geometry' not in gdf_to_plot.columns: return None
    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        if 'Cluster' not in gdf_to_plot.columns or gdf_to_plot['Cluster'].isna().all():
            gdf_to_plot.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.3)
            ax.set_title("Peta Dasar Wilayah (Cluster Tidak Tersedia)", fontsize=10)
        else:
            clusters_map = sorted([c for c in gdf_to_plot['Cluster'].unique() if pd.notna(c) and c != -1])
            color_map_static = get_cluster_color_map(clusters_map)

            # Plot NaN
            gdf_to_plot[gdf_to_plot['Cluster'].isna()].plot(
                color=color_map_static.get(None, 'white'), # Warna NaN
                ax=ax, edgecolor='lightgray', linewidth=0.2
            )
            # Plot Noise (-1)
            gdf_to_plot[gdf_to_plot['Cluster'] == -1].plot(
                color=color_map_static.get(-1, 'lightgrey'), # Warna Noise
                ax=ax, edgecolor='darkgrey', linewidth=0.2
            )
            # Plot Cluster Valid (loop agar warna pas)
            for cluster_id in clusters_map:
                color = color_map_static.get(cluster_id, 'black')
                gdf_to_plot[gdf_to_plot['Cluster'] == cluster_id].plot(
                    color=color, ax=ax, edgecolor='black', linewidth=0.2, label=f"Cluster {cluster_id}"
                )

            ax.set_title("Peta Persebaran Cluster (Statis)", fontsize=10)
            ax.legend(title="Cluster", loc='lower right', fontsize='x-small') # Andalkan legend dari .plot

        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=150); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"[PDF] Error rendering peta statis ke buffer: {e}")
        if fig is not None and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def render_kmeans_helpers(k_search_data):
    """
    Render tabel justifikasi pencarian K-Optimal untuk K-Means.
    """
    if k_search_data is None or k_search_data.empty:
        return

    st.subheader("PENCARIAN K-OPTIMAL (K-MEANS)", help="Tabel ini menunjukkan hasil Skor Silhouette untuk setiap nilai K yang diuji. Sistem memilih K dengan skor tertinggi.")
    
    try:
        df_display = k_search_data.copy()
        
        best_k_index = df_display['Silhouette'].idxmax()
        
        def style_row(row):
            styles = [''] * len(row)
            if row.name == best_k_index:
                styles = ['background-color: #e0f7fa; font-weight: bold;'] * len(row) # Sorot baris terbaik
            return styles

        df_display['Silhouette'] = df_display['Silhouette'].apply(lambda x: f"{x:.4f}" if x != -1 else "N/A (Gagal)")
        
        st.dataframe(
            df_display.style.apply(style_row, axis=1), 
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Gagal render tabel K-Optimal: {e}")
        st.dataframe(k_search_data, use_container_width=True)

def render_dbscan_helpers(elbow_data, elbow_minpts, minpts_plot_data, elbow_knee):
    """
    Render K-distance (Elbow) plot dan Sil vs MinPts plot untuk DBSCAN.
    """
    
    if elbow_data is None or minpts_plot_data is None:
        return
    
    st.divider()

    st.subheader("ELBOW & MINPTS PLOT (DBSCAN)", help="Visualisasi ini membantu dalam menentukan parameter DBSCAN yang optimal, yaitu Epsilon dan MinPts. Plot K-distance (Elbow) digunakan untuk memilih nilai Epsilon, sedangkan plot Silhouette vs MinPts membantu dalam memilih nilai MinPts yang sesuai.")
    col1, col2 = st.columns(2)

    with col1:
        try:
            # --- 1. Elbow Plot (K-distance) ---
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=np.arange(len(elbow_data)),
                y=elbow_data,
                mode='lines',
                name=f'k-distance (k={elbow_minpts})'
            ))
            
            # Tambahkan garis siku jika ditemukan
            if elbow_knee and elbow_knee[0] is not None and elbow_knee[1] is not None:
                knee_x, knee_y = elbow_knee
                if knee_y > 0:
                    fig_elbow.add_vline(x=knee_x, line_dash="dash", line_color="red")
                    fig_elbow.add_hline(y=knee_y, line_dash="dash", line_color="red")
                    fig_elbow.add_annotation(x=knee_x, y=knee_y, text=f"Siku (Eps) ≈ {knee_y:.2f}", showarrow=True, arrowhead=1, ay=-30)

            fig_elbow.update_layout(
                title=f"Plot Siku (K-Distance) untuk MinPts={elbow_minpts}",
                xaxis_title="Titik (diurutkan berdasarkan jarak)",
                yaxis_title=f"Jarak Tetangga ke-{elbow_minpts} (Epsilon)",
                height=400
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal render Elbow Plot: {e}")

    with col2:
        try:
            # --- 2. Silhouette vs MinPts Plot ---
            fig_minpts = go.Figure()
            fig_minpts.add_trace(go.Scatter(
                x=minpts_plot_data['MinPts'],
                y=minpts_plot_data['Silhouette'],
                mode='lines+markers',
                text=minpts_plot_data.apply(lambda row: f"Eps: {row['Eps_Found']:.2f}", axis=1),
                hovertemplate='MinPts: %{x}<br>Silhouette: %{y:.4f}<br>Eps terkait: %{text}'
            ))
            
            # Tandai skor terbaik
            if not minpts_plot_data.empty:
                best_idx = minpts_plot_data['Silhouette'].idxmax()
                best_row = minpts_plot_data.loc[best_idx]
                
                fig_minpts.add_vline(x=best_row['MinPts'], line_dash="dash", line_color="green")
                fig_minpts.add_annotation(
                    x=best_row['MinPts'], 
                    y=best_row['Silhouette'], 
                    text=f"Terbaik: {best_row['Silhouette']:.4f} (MinPts={best_row['MinPts']})",
                    showarrow=True,
                    arrowhead=1,
                    ay=-30 # offset anotasi
                )

            fig_minpts.update_layout(
                title="Skor Silhouette vs. MinPts (Rentang: D+1 s.d. 20)",
                xaxis_title="MinPts",
                yaxis_title="Silhouette Score",
                height=400,
                xaxis=dict(tickmode='linear', dtick=1)
            )
            st.plotly_chart(fig_minpts, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal render MinPts Plot: {e}")