import streamlit as st
import io
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.metrics import silhouette_samples
import numpy as np
import folium
import seaborn as sns
import matplotlib.colors as mcolors

def get_cluster_color_map(cluster_ids):
    """Mendapatkan peta warna untuk cluster berdasarkan cluster_ids."""
    color_map = {}
    
    # Filter cluster_ids valid (bukan NaN atau -1)
    valid_cluster_ids = sorted([int(c) for c in cluster_ids if pd.notna(c) and c != -1])
    n_colors = len(valid_cluster_ids)
    try:
        # Ambil colormap 'tab10' dari Matplotlib untuk konsistensi warna
        cmap = plt.colormaps.get('tab10')
        # Assign warna ke setiap cluster valid dari colormap tersebut
        # Langkah: normalisasi indeks warna ke [0, 1], ambil warna RGBA, konversi ke HEX
        for i, cluster_id in enumerate(valid_cluster_ids):
            norm_index = float(i) / (n_colors -1) if n_colors > 1 else 0.5
            rgba_color = cmap(norm_index)
            color_map[cluster_id] = mcolors.to_hex(rgba_color)
    except Exception as e:
        st.warning(f"Gagal get cmap 'tab10' ({e}), fallback ke Plotly.")
        plotly_colors = px.colors.qualitative.Plotly
        for i, cluster_id in enumerate(valid_cluster_ids):
            color_map[cluster_id] = plotly_colors[i % len(plotly_colors)]
            
    # Warna untuk noise (-1) dan NaN
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
        
        help_sil = "Semakin tinggi nilai Silhouette, data dalam cluster semakin sesuai dengan tempatnya, artinya cluster lumayan benar."
        help_dbi = "Semakin kecil nilai DBI, cluster lebih kompak dan terpisah, menandakan sedikit atau tidak ada tumpang tindih."
        help_time = "Semakin kecil waktu pemrosesan, clustering selesai lebih cepat."
        
        met_col1.metric("Silhouette", f"{sil_score:.3f}" if sil_score is not None else "N/A", help=help_sil)
        met_col2.metric("DBI", f"{dbi_score:.3f}" if dbi_score is not None else "N/A", help=help_dbi)
        met_col3.metric("Waktu", f"{time_sec:.2f} s" if time_sec is not None else "N/A", help=help_time)
        
    else:
        # Placeholder jika tidak ada skor
        st.container(height=80)
        
    # ============================================================
    st.divider()
    # ============================================================
    
    # Panggil fungsi render_silhouette_plot (Silhouette Plot)
    render_silhouette_plot(data_for_clustering, hasil_data, scores)

def create_folium_map(gdf_merged, key_column='WADMKK'):
    """
    Membuat peta interaktif Folium berdasarkan GeoDataFrame, dengan tooltip dan legenda
    """
    
    # Cek validitas data:
    # Jika gdf_merged tidak valid, kembalikan None
    if gdf_merged is None or gdf_merged.empty or 'geometry' not in gdf_merged.columns:
        st.warning("Data geospasial tidak valid untuk membuat peta."); return None
    # Jika kolom 'Cluster' atau key_column tidak ada, kembalikan None
    if 'Cluster' not in gdf_merged.columns:
        st.warning("Kolom 'Cluster' tidak ditemukan di data geospasial."); return None
    # Jika key_column tidak ada, coba fallback ke 'kab_kota'
    if key_column not in gdf_merged.columns:
        st.warning(f"Kolom kunci '{key_column}' tidak ditemukan."); key_column = 'kab_kota';
        if key_column not in gdf_merged.columns: st.error("Tidak dapat menemukan kolom nama wilayah."); return None

    # Default fields/aliases untuk tooltip
    fields_for_tooltip = [key_column, 'Cluster']
    aliases_for_tooltip = ['Wilayah:', 'Cluster:']
    # Cek jika 'prov' ada dan update
    if 'prov' in gdf_merged.columns:
        # Tooltip dengan provinsi
        fields_for_tooltip = ['prov', key_column, 'Cluster']
        # Aliases
        aliases_for_tooltip = ['Provinsi:', 'Wilayah:', 'Cluster:']
    else:
        st.warning("Kolom 'prov' tidak ditemukan di data gabungan untuk tooltip.")

    try:
        # Koordinat pusat Indonesia
        map_center = [-2.5489, 118.0149]; zoom_level = 5
        # Buat peta Folium
        m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="cartodbpositron")

        # Mengurutkan cluster yang valid untuk konsistensi warna
        clusters_valid = sorted([c for c in gdf_merged['Cluster'].unique() if pd.notna(c) and c != -1])
        # Memanggil fungsi get_cluster_color_map untuk mendapatkan peta warna
        color_dict = get_cluster_color_map(clusters_valid)
        # Fungsi untuk mendapatkan warna berdasarkan cluster, default ke abu-abu jika tidak ditemukan
        get_color = lambda cluster_id: color_dict.get(cluster_id, '#808080')
        
        # Buat GeoJson dengan style dan tooltip
        geojson = folium.GeoJson(
            gdf_merged,
            style_function=lambda feature: {
                'fillColor': get_color(feature['properties'].get('Cluster')), # Gunakan get_color
                'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7,
            },
            # Highlight saat hover
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

        # Buat legenda kustom
        # Jika ada cluster valid, buat legend
        if clusters_valid:
            legend_html = '''
                <div style="position:fixed; bottom:50px; right:50px; width:150px; height:auto; 
                border:2px solid grey; z-index:9999; font-size:12px; background-color:white; 
                padding:10px; opacity:0.9;"><b>Legenda Cluster</b><br>
            '''
            # Dalam setiap cluster valid, tambahkan ke legend
            for cluster_id in clusters_valid:
                # Dapatkan warna dari dict, black jika tidak ditemukan
                color = color_dict.get(cluster_id, 'black')
                # Tambah entri legenda untuk cluster
                legend_html += f'&nbsp; <i style="background:{color}; width:15px; height:15px; display:inline-block; margin-right:5px; border: 1px solid grey;"></i> Cluster {int(cluster_id)}<br>'
            # Cek jika ada noise (-1) atau NaN
            has_noise = -1 in gdf_merged['Cluster'].unique()
            has_nan = gdf_merged['Cluster'].isnull().any()
            
            # Jika ada noise/NaN, tambahkan entri legenda untuk noise/NaN
            if has_noise or has_nan:
                color_na = color_dict.get(-1, 'lightgrey')
                legend_html += f'&nbsp; <i style="background:{color_na}; width:15px; height:15px; display:inline-block; margin-right:5px; border: 1px solid grey;"></i> Noise / N/A<br>'
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))
            
        return m
    except Exception as e:
        # DEBUG
        st.error(f"Gagal membuat peta Folium: {e}")
        return None

def render_boxplot(df_hasil, data_for_clustering=None):
    """Membuat dan menampilkan boxplot interaktif menggunakan pd.melt."""
    # Jika df_hasil tidak valid, tampilkan info dan keluar
    if df_hasil is None or df_hasil.empty: st.info("Data tidak valid."); return
    # Jika kolom 'Cluster' tidak ada, tampilkan peringatan dan keluar
    if 'Cluster' not in df_hasil.columns: st.warning("Kolom 'Cluster' tidak ditemukan."); return
    # Jika tidak ada cluster valid, tampilkan info dan keluar
    if df_hasil['Cluster'].nunique() < 1 or (df_hasil['Cluster'].nunique() == 1 and df_hasil['Cluster'].unique()[0] == -1): st.info("Tidak ada cluster valid."); return

    try:
        # Jika data_for_clustering disediakan, gunakan kolomnya sebagai value_vars untuk dimelt
        if data_for_clustering is not None and not data_for_clustering.empty:
            value_vars = data_for_clustering.columns.tolist()
        else:
            # Jika tidak, gunakan semua kolom numerik kecuali 'ID' dan 'Cluster' untuk dimelt
            potential_value_vars = df_hasil.select_dtypes(include=np.number).columns.tolist()
            exclude_cols = ['ID', 'Cluster']
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
    if data_for_clustering is None or hasil_data is None or scores is None: st.info("Data tidak valid."); return
    
    # Mengambil label unik
    labels = hasil_data['Cluster']
    unique_labels = labels.unique()
    
    # Jumlah cluster valid berdasarkan label unik
    n_clusters_valid = len([l for l in unique_labels if l != -1 and pd.notna(l)])
    
    # Jika kurang dari 2 cluster valid, tampilkan peringatan dan keluar
    if n_clusters_valid < 2: st.warning("Silhouette plot memerlukan minimal 2 cluster valid."); return
    
    # Mengambil nilai silhouette rata-rata. Jika tidak ada, tampilkan peringatan dan keluar
    silhouette_avg = scores.get('silhouette')
    if silhouette_avg is None: st.warning("Silhouette score tidak ditemukan."); return
    
    
    try:
        sample_silhouette_values = silhouette_samples(data_for_clustering, labels)
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
    """Membuat pair plot Seaborn dengan warna konsisten."""
    
    st.subheader("SEBARAN DATA (PAIR PLOT SEABORN)")
    
    # Jika df_hasil tidak valid, tampilkan info dan keluar
    if df_hasil is None or df_hasil.empty: st.info("Data tidak valid."); return
    # Jika kolom 'Cluster' tidak ada, tampilkan peringatan dan keluar
    if 'Cluster' not in df_hasil.columns: st.warning("Kolom 'Cluster' tidak ditemukan."); return
    # Jika data_for_clustering tidak valid, tampilkan info dan keluar
    if data_for_clustering is None or data_for_clustering.empty: st.info("Data untuk clustering tidak valid."); return
    
    # Mendapat variabel untuk ditampilkan
    scatter_vars = data_for_clustering.columns.tolist()
    if len(scatter_vars) < 2: st.info("Tidak cukup variabel untuk ditampilkan."); return
    
    try:
        # Persiapan data untuk plot
        # Pilih kolom yang ada di df_hasil
        cols_to_plot = [col for col in scatter_vars if col in df_hasil.columns] + ['Cluster', 'kab_kota']
        # Jika kurang dari 3 kolom (2 variabel + Cluster), tampilkan error dan keluar
        if len(cols_to_plot) < 3: st.error("Tidak cukup kolom untuk ditampilkan."); return
        # Mengambil data untuk plot
        plot_data = df_hasil[cols_to_plot].copy()
        # Hapus baris dengan nilai NaN pada variabel scatter
        plot_data.dropna(subset=scatter_vars, inplace=True)
        # Jika plot_data kosong setelah dropna, tampilkan peringatan dan keluar
        if plot_data.empty: st.warning("Tidak ada data untuk ditampilkan."); return
        # Pastikan kolom 'Cluster' bertipe kategori untuk pewarnaan
        plot_data['Cluster'] = plot_data['Cluster'].astype('category')
        # Mendapat daftar cluster valid
        cluster_categories_scatter = sorted([c for c in plot_data['Cluster'].unique() if c != -1 and pd.notna(c)])
        
        # Memanggil fungsi get_cluster_color_map untuk mendapatkan peta warna
        color_map_scatter = get_cluster_color_map(cluster_categories_scatter)
        try:
            # Membuat palette dictionary untuk Seaborn
            seaborn_palette = {k: v for k, v in color_map_scatter.items() if k != -1 and k is not None}
            # Jika ada noise (-1), tambahkan ke palette
            if -1 in plot_data['Cluster'].cat.categories:
                seaborn_palette[-1] = color_map_scatter.get(-1, 'lightgrey')
        except:
            seaborn_palette = {str(k): v for k, v in color_map_scatter.items() if k != -1 and k is not None}
            if '-1' in plot_data['Cluster'].cat.categories.astype(str):
                seaborn_palette['-1'] = color_map_scatter.get(-1, 'lightgrey')
        
        # Set tema Seaborn dan buat pair plot
        sns.set_theme(style="ticks")
        pair_plot = sns.pairplot(
            plot_data, vars=scatter_vars,
            # pewarnaan berdasarkan cluster
            hue="Cluster",
            # diagonal plot jenis KDE
            diag_kind="kde",
            # kustom marker dan style
            plot_kws={'s': 25, 'alpha': 0.8, 'edgecolor': 'w', 'linewidth': 0.5},
            corner=False,
            # pewarnaan dengan palette
            palette=seaborn_palette
        )
        # Set judul utama dan tampilkan plot
        pair_plot.fig.suptitle(f"Pair Plot Seaborn ({', '.join(scatter_vars)}) - Data Normalisasi", y=1.02)
        st.pyplot(pair_plot.fig, clear_figure=True)
        plt.close(pair_plot.fig)
        st.caption("Plot ini menggunakan data yang sudah dinormalisasi.")
    except Exception as e:
        st.error(f"Gagal membuat Seaborn pair plot: {e}")

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

def render_silhouette_plot_to_buffer(data_for_clustering, hasil_data, scores):
    fig = None
    try:
        labels = hasil_data['Cluster']; unique_labels = labels.unique()
        n_clusters_valid = len([l for l in unique_labels if l != -1 and pd.notna(l)])
        if n_clusters_valid < 2: return None
        silhouette_avg = scores.get('silhouette');
        if silhouette_avg is None: return None
        sample_silhouette_values = silhouette_samples(data_for_clustering, labels)

        fig, ax = plt.subplots(figsize=(6, 4)); y_lower = 10
        cluster_labels_sorted = sorted([label for label in unique_labels if label != -1 and pd.notna(label)])
        color_map_sil_buf = get_cluster_color_map(cluster_labels_sorted)

        for i in cluster_labels_sorted:
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]; ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]; y_upper = y_lower + size_cluster_i
            color = color_map_sil_buf.get(i, 'black')
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=8); y_lower = y_upper + 10
        ax.set_title("Silhouette Plot", fontsize=10); ax.set_xlabel("Silhouette Coefficient", fontsize=9); ax.set_ylabel("Cluster", fontsize=9)
        ax.axvline(x=silhouette_avg, color="red", linestyle="--"); ax.set_yticks([])
        ax.set_xticks(np.arange(-0.1, 1.1, 0.2)); ax.set_xlim([-0.1, 1.0]); ax.tick_params(axis='x', labelsize=8)
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=150); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error rendering silhouette plot to buffer: {e}")
        if fig is not None and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def render_boxplot_to_buffer(df_hasil, data_for_clustering=None):
    """Render boxplot ke buffer BytesIO dengan warna konsisten."""
    try:
        exclude_cols_box = ['No', 'Cluster']; value_vars = data_for_clustering.columns.tolist() if data_for_clustering is not None else [col for col in df_hasil.select_dtypes(include=np.number).columns if col not in exclude_cols_box and '_' in col]
        if not value_vars: return None
        id_vars = ['kab_kota', 'Cluster']; df_hasil_melt = df_hasil.copy(); df_hasil_melt['kab_kota'] = df_hasil_melt['kab_kota'].astype(str)
        df_long = pd.melt(df_hasil_melt, id_vars=id_vars, value_vars=value_vars, var_name='Variable_Tahun', value_name='Nilai')
        try:
            split_data = df_long['Variable_Tahun'].str.extract(r'^(.*?)_(\d{4})$');
            if split_data.isnull().all().all(): split_data = df_long['Variable_Tahun'].str.rsplit('_', n=1, expand=True)
            df_long['Variabel'] = split_data[0]; df_long['Tahun'] = split_data[1]; df_long.dropna(subset=['Variabel', 'Tahun'], inplace=True)
            if df_long.empty: return None
            unique_years = sorted(df_long['Tahun'].unique()); df_long['Tahun'] = pd.Categorical(df_long['Tahun'], categories=unique_years, ordered=True)
        except Exception: return None # Gagal parse
        unique_clusters_valid = sorted([c for c in df_long['Cluster'].unique() if pd.notna(c) and c != -1])
        df_long['Cluster_Str'] = df_long['Cluster'].astype(str) # String untuk Plotly
        cluster_categories_str = sorted([str(c) for c in unique_clusters_valid], key=lambda x: int(x))
        if '-1' in df_long['Cluster_Str'].unique(): cluster_categories_str.append('-1')

        color_map_plotly_buf = get_cluster_color_map(unique_clusters_valid)
        color_discrete_map_box_buf = {str(k): v for k, v in color_map_plotly_buf.items()}
        if '-1' in df_long['Cluster_Str'].unique(): color_discrete_map_box_buf['-1'] = color_map_plotly_buf.get(-1, 'lightgrey')

        fig_box = px.box(
            df_long, x='Tahun', y='Nilai', color='Cluster_Str', facet_col='Variabel',
            title="Distribusi Nilai Variabel per Cluster dan Tahun",
            category_orders={"Tahun": unique_years, "Cluster_Str": cluster_categories_str},
            color_discrete_map=color_discrete_map_box_buf,
            facet_col_wrap=2, facet_col_spacing=0.05, facet_row_spacing=0.1
        )
        fig_box.update_yaxes(matches=None, showticklabels=True, title_text="")
        fig_box.update_xaxes(title_text="Tahun")
        fig_box.update_layout(height=400*((len(value_vars)+1)//2), boxmode='group', font_size=8, title_font_size=10, legend_title_text='Cluster')
        fig_box.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        buf = io.BytesIO()
        fig_box.write_image(buf, format='png', scale=2, engine='kaleido')
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error rendering boxplot to buffer: {e}")
        return None