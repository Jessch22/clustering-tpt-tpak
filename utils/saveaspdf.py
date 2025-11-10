import streamlit as st
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib.patches import Rectangle

from modules.plot import get_cluster_color_map



# ====================================================
# === CUSTOM PDF CLASS
# ====================================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Laporan Analisis Clustering TPT & TPAK", 0, 1, 'C')
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 13)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(3)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln()

    def add_dataframe_to_pdf(self, df, max_rows=50, cols_to_show=None):
        """Menampilkan tabel kecil hasil clustering di PDF"""
        if df is None or df.empty:
            self.set_font('Arial', 'I', 9)
            self.multi_cell(0, 6, "(Tidak ada data untuk ditampilkan)")
            return

        df_to_show = df.head(max_rows)
        if cols_to_show:
            cols_exist = [col for col in cols_to_show if col in df_to_show.columns]
            df_to_show = df_to_show[cols_exist]

        self.set_font('Courier', '', 5.5)
        table_str = df_to_show.to_string(index=False)
        self.multi_cell(0, 3, table_str)
        self.ln(5)
        self.set_font('Arial', '', 11)


# ====================================================
# === UTILITAS: KONVERSI PLOT KE BUFFER UNTUK PDF
# ====================================================
def render_static_map_to_buffer(gdf_hasil):
    """Render GeoDataFrame ke buffer gambar untuk PDF"""
    if gdf_hasil is None or gdf_hasil.empty or 'geometry' not in gdf_hasil.columns: 
        return None
    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        legend_handles = []
        
        # Cek jika tidak ada cluster valid
        if 'Cluster' not in gdf_hasil.columns or gdf_hasil['Cluster'].isna().all():
            gdf_hasil.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.3)
            ax.set_title("Peta Dasar Wilayah (Cluster Tidak Tersedia)", fontsize=10)
        
        else:
            # Dapatkan warna konsisten
            all_clusters_ids = gdf_hasil['Cluster'].unique()
            clusters_map_valid = sorted([c for c in all_clusters_ids if pd.notna(c) and c != -1])
            color_map_static = get_cluster_color_map(all_clusters_ids)

            # Plot wilayah NaN (jika ada)
            gdf_nan = gdf_hasil[gdf_hasil['Cluster'].isna()]
            if not gdf_nan.empty:
                color_nan = color_map_static.get(np.nan, '#D3D3D3')
                gdf_nan.plot(
                    color=color_nan,
                    ax=ax, edgecolor='lightgray', linewidth=0.2
                )
                legend_handles.append(Rectangle((0, 0), 1, 1, fc=color_nan, ec='lightgray', lw=0.2, label='N/A'))

            # --- Plot Noise ---
            gdf_noise = gdf_hasil[gdf_hasil['Cluster'] == -1]
            if not gdf_noise.empty:
                color_noise = color_map_static.get(-1, '#D3D3D3')
                gdf_noise.plot(
                    color=color_noise,
                    ax=ax, edgecolor='darkgrey', linewidth=0.2
                )
                legend_handles.append(Rectangle((0, 0), 1, 1, fc=color_noise, ec='darkgrey', lw=0.2, label='Noise / -1'))
            

            # --- Plot Cluster Valid ---
            for cluster_id in clusters_map_valid:
                color = color_map_static.get(cluster_id, 'black')
                gdf_hasil[gdf_hasil['Cluster'] == cluster_id].plot(
                    color=color, ax=ax, edgecolor='black', linewidth=0.2, label=f'Cluster {cluster_id}'
                )
                legend_handles.append(Rectangle((0, 0), 1, 1, fc=color, ec='black', lw=0.2, label=f'Cluster {cluster_id}'))
                
            ax.set_title("Peta Persebaran Cluster (Statis)", fontsize=10)
            if legend_handles:
                ax.legend(handles=legend_handles, title="Cluster", loc='best', 
                            fontsize='small', frameon=True, facecolor='white', framealpha=0.8)

        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        print(f"[PDF] Error rendering peta statis ke buffer: {e}")
        if fig is not None and plt.fignum_exists(fig.number): 
            plt.close(fig)
        return None


def render_silhouette_to_buffer(hasil_data, data_for_clustering):
    """Render silhouette plot"""
    try:
        if data_for_clustering is None or hasil_data is None or 'Cluster' not in hasil_data.columns:
            return None
        
        labels = hasil_data['Cluster']
        unique_clusters = sorted([c for c in labels.unique() if pd.notna(c)]) # Ambil semua cluster non-NaN
        
        # Perlu setidaknya 2 cluster (termasuk noise -1 jika ada) untuk silhouette_samples
        if len(unique_clusters) < 2:
            print("Silhouette plot memerlukan setidaknya 2 cluster.")
            return None
            
        # Hitung silhouette samples
        silhouette_vals = silhouette_samples(data_for_clustering, labels)
        avg_score = silhouette_vals.mean()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        y_lower = 10
        
        # Dapatkan peta warna konsisten
        color_map = get_cluster_color_map(unique_clusters)
        legend_handles = []
        
        for i, cluster in enumerate(unique_clusters):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            
            size_cluster = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster
            
            # Gunakan warna dari color_map
            color = color_map.get(cluster, 'black') 
            
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            proxy_patch = Rectangle((0, 0), 1, 1, fc=color, ec=color, alpha=0.7, 
                                    label=f'Cluster {cluster}')
            legend_handles.append(proxy_patch)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster, f'C{cluster}', fontsize=10, fontweight='bold')
            y_lower = y_upper + 10
        
        ax.set_title("Silhouette Plot", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Silhouette Coefficient", fontsize=12)
        ax.set_ylabel("Cluster", fontsize=12)
        avg_line = ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=2, 
                                label=f'Rata-rata: {avg_score:.3f}')
        legend_handles.append(avg_line)
        ax.legend(handles=legend_handles, fontsize=10, loc='best')
        
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Gagal render silhouette plot: {e}")
        return None


def render_boxplot_to_buffer(hasil_data, data_for_clustering):
    """Render boxplot seperti di web"""
    try:
        if data_for_clustering is None or hasil_data is None:
            return None
            
        n_cols = len(data_for_clustering.columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        
        if n_cols == 1:
            axes = [axes]
            
        unique_clusters = sorted(hasil_data['Cluster'].unique())
        color_map = get_cluster_color_map(unique_clusters)
        colors = [color_map.get(c, 'black') for c in unique_clusters]
            
        for idx, col in enumerate(data_for_clustering.columns):
            cluster_data = []
            cluster_labels = []
            
            for cluster in unique_clusters:
                cluster_values = hasil_data[hasil_data['Cluster'] == cluster][col].values
                cluster_values = cluster_values[~np.isnan(cluster_values)]
                cluster_data.append(cluster_values)
                cluster_labels.append(f'C{cluster}')
            
            if not any(len(cd) > 0 for cd in cluster_data):
                axes[idx].text(0.5, 0.5, 'Tidak ada data valid', horizontalalignment='center', verticalalignment='center', transform=axes[idx].transAxes, color='red')
                axes[idx].set_title(f'{col}', fontsize=13, fontweight='bold')
                continue

            bp = axes[idx].boxplot(cluster_data, labels=cluster_labels, patch_artist=True, showfliers=False) 
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[idx].set_title(f'{col}', fontsize=13, fontweight='bold')
            axes[idx].set_xlabel('Cluster', fontsize=11)
            axes[idx].set_ylabel('Nilai', fontsize=11)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle("Distribusi Nilai per Cluster (Box Plot)", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Gagal render boxplot: {e}")
        return None


# def render_scatter_to_buffer(hasil_data, data_for_clustering):
#     """Render scatter plot seperti di web"""
#     try:
#         if data_for_clustering is None or hasil_data is None:
#             return None
            
#         n_cols = len(data_for_clustering.columns)
        
#         if n_cols >= 2:
#             # Scatter matrix jika ada 2+ kolom - ukuran dikurangi agar muat
#             fig_size = min(10, 3.5 * n_cols)  # Batasi ukuran maksimal
#             fig, axes = plt.subplots(n_cols, n_cols, figsize=(fig_size, fig_size))
            
#             for i in range(n_cols):
#                 for j in range(n_cols):
#                     ax = axes[i, j] if n_cols > 1 else axes
                    
#                     if i == j:
#                         # Histogram di diagonal
#                         for cluster in sorted(hasil_data['Cluster'].unique()):
#                             cluster_data = hasil_data[hasil_data['Cluster'] == cluster][data_for_clustering.columns[i]]
#                             ax.hist(cluster_data, alpha=0.5, label=f'C{cluster}', bins=15)
#                         ax.set_ylabel('Frekuensi', fontsize=8)
#                         if i == 0:
#                             ax.legend(loc='upper right', fontsize=7)
#                     else:
#                         # Scatter plot
#                         scatter = ax.scatter(
#                             hasil_data[data_for_clustering.columns[j]],
#                             hasil_data[data_for_clustering.columns[i]],
#                             c=hasil_data['Cluster'],
#                             cmap='tab10',
#                             alpha=0.6,
#                             s=15,
#                             edgecolors='black',
#                             linewidth=0.3
#                         )
                    
#                     # Label axes dengan font lebih kecil
#                     if j == 0:
#                         ax.set_ylabel(data_for_clustering.columns[i], fontsize=8)
#                     else:
#                         ax.set_ylabel('')
                    
#                     if i == n_cols - 1:
#                         ax.set_xlabel(data_for_clustering.columns[j], fontsize=8)
#                     else:
#                         ax.set_xlabel('')
                    
#                     ax.grid(True, alpha=0.3)
#                     ax.tick_params(labelsize=7)
            
#             plt.suptitle("Visualisasi Scatter Plot Matrix", fontsize=12, fontweight='bold', y=0.995)
#             plt.tight_layout()
            
#         else:
#             # Jika hanya 1 kolom, buat histogram
#             fig, ax = plt.subplots(figsize=(10, 7))
#             for cluster in sorted(hasil_data['Cluster'].unique()):
#                 cluster_data = hasil_data[hasil_data['Cluster'] == cluster][data_for_clustering.columns[0]]
#                 ax.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', bins=20)
#             ax.set_xlabel(data_for_clustering.columns[0], fontsize=12)
#             ax.set_ylabel('Frekuensi', fontsize=12)
#             ax.set_title('Distribusi Data per Cluster', fontsize=14, fontweight='bold')
#             ax.legend(fontsize=10)
#             ax.grid(True, alpha=0.3)
            
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
#         plt.close(fig)
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"Gagal render scatter plot: {e}")
#         import traceback
#         st.error(traceback.format_exc())
#         return None


# ====================================================
# === FUNGSI UTAMA: GENERATE PDF REPORT
# ====================================================
def generate_pdf_report():
    """Membuat laporan PDF komprehensif berdasarkan hasil analisis di session_state."""

    var = st.session_state.get("var", "N/A")
    tahun_pilihan = st.session_state.get("tahun_pilihan", ("N/A", "N/A"))
    metode = st.session_state.get("metode_pilihan", "N/A")
    params_dict = st.session_state.get("params", {})
    scores = st.session_state.get("scores")
    hasil_data = st.session_state.get("hasil_data")
    data_for_clustering = st.session_state.get("data_for_clustering")
    gdf_hasil = st.session_state.get("gdf_hasil")

    if hasil_data is None:
        st.warning("Data hasil analisis tidak ditemukan. PDF tidak dapat dibuat.")
        return None

    params_str = (
        ", ".join([f"{k}={v}" for k, v in params_dict.items()]) if params_dict else "N/A"
    )

    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.alias_nb_pages()
    
    # === HALAMAN 1: PARAMETER & METRIK ===
    pdf.add_page()
    pdf.chapter_title("1. Parameter Analisis")
    input_text = (
        f"Variabel yang Digunakan: {var}\n"
        f"Rentang Tahun Analisis: {tahun_pilihan[0]} - {tahun_pilihan[1]}\n"
        f"Metode Clustering: {metode}\n"
        f"Parameter: {params_str}"
    )
    pdf.chapter_body(input_text)

    pdf.ln(5)
    pdf.chapter_title("2. Hasil Metrik Evaluasi")
    if scores:
        sil = scores.get("silhouette")
        dbi = scores.get("dbi")
        time_val = scores.get("time_sec")
        
        if sil > 0.7:
            interpretasiS = "Struktur Kuat"
        elif sil > 0.5:
            interpretasiS = "Struktur Sedang"
        elif sil > 0.25:
            interpretasiS = "Struktur Lemah"
        else:
            interpretasiS = "Tidak ada Struktur"
        metric_text = (
            f"Silhouette Score: {sil:.4f}\n"
            f"   - Interpretasi: {interpretasiS}\n\n"
            f"Davies-Bouldin Index: {dbi:.4f}\n"
            f"   - Interpretasi: {'Baik' if dbi < 1.0 else 'Cukup' if dbi < 2.0 else 'Kurang Baik'}\n\n"
            f"Waktu Komputasi: {time_val:.2f} detik\n"
            f"Jumlah Total Data: {len(hasil_data)} wilayah\n"
            f"Jumlah Cluster Terbentuk: {hasil_data['Cluster'].nunique()}"
        )
        pdf.chapter_body(metric_text)
    else:
        pdf.chapter_body("(Skor metrik tidak tersedia)")

    # === HALAMAN 2: SILHOUETTE PLOT (Full Page) ===
    pdf.add_page()
    pdf.chapter_title("3. Silhouette Plot")
    sil_buf = render_silhouette_to_buffer(hasil_data, data_for_clustering)
    if sil_buf:
        # Full page silhouette plot
        pdf.image(sil_buf, x=50, y=pdf.get_y(), w=200)
    else:
        pdf.chapter_body("(Silhouette plot tidak tersedia)")

    # === HALAMAN 3: PETA PERSEBARAN CLUSTER (Full Page) ===
    pdf.add_page()
    # === HALAMAN 3: PETA PERSEBARAN CLUSTER (Full Page) ===
    pdf.add_page()
    pdf.chapter_title("4. Peta Persebaran Cluster Wilayah")
    if gdf_hasil is not None:
        map_buf = render_static_map_to_buffer(gdf_hasil)
        if map_buf:
            # Full page map
            pdf.image(map_buf, x=10, y=pdf.get_y(), w=277)
        else:
            pdf.chapter_body("(Gagal membuat peta statis)")
    else:
        pdf.chapter_body("(Data peta tidak tersedia)")

    # === HALAMAN 4: DISTRIBUSI CLUSTER (BOX PLOT - Full Page) ===
    pdf.add_page()
    pdf.chapter_title("5. Distribusi Nilai per Cluster (Box Plot)")
    box_buf = render_boxplot_to_buffer(hasil_data, data_for_clustering)
    if box_buf:
        # Full page boxplot
        pdf.image(box_buf, x=10, y=pdf.get_y(), w=277)
    else:
        pdf.chapter_body("(Box plot tidak tersedia)")

    # # === HALAMAN 5: SCATTER PLOT MATRIX (Full Page) ===
    # pdf.add_page()
    # pdf.chapter_title("6. Visualisasi Scatter Plot Matrix")
    # scatter_buf = render_scatter_to_buffer(hasil_data, data_for_clustering)
    # if scatter_buf:
    #     # Ukuran disesuaikan agar muat di halaman
    #     pdf.image(scatter_buf, x=20, y=pdf.get_y(), w=200)
    # else:
    #     pdf.chapter_body("(Scatter plot tidak tersedia)")

    # === HALAMAN 5: STATISTIK DESKRIPTIF PER CLUSTER ===
    pdf.add_page()
    pdf.chapter_title("7. Statistik Deskriptif per Cluster")
    
    try:
        cluster_stats = hasil_data.groupby('Cluster').agg({
            col: ['mean', 'std', 'min', 'max'] 
            for col in data_for_clustering.columns
        }).round(2)
        
        cluster_counts = hasil_data['Cluster'].value_counts().sort_index()
        
        for cluster in sorted(hasil_data['Cluster'].unique()):
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, f"Cluster {cluster} (Jumlah: {cluster_counts[cluster]} wilayah)", 0, 1, 'L')
            pdf.set_font('Courier', '', 8)
            
            stats_text = ""
            for col in data_for_clustering.columns:
                mean_val = cluster_stats.loc[cluster, (col, 'mean')]
                std_val = cluster_stats.loc[cluster, (col, 'std')]
                min_val = cluster_stats.loc[cluster, (col, 'min')]
                max_val = cluster_stats.loc[cluster, (col, 'max')]
                
                stats_text += f"  {col}:\n"
                stats_text += f"    Mean: {mean_val:.2f}, Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}\n"
            
            pdf.multi_cell(0, 5, stats_text)
            pdf.ln(3)
    except Exception as e:
        pdf.chapter_body(f"(Gagal menghitung statistik: {e})")

    # === HALAMAN 7: TABEL HASIL ===
    pdf.add_page()
    pdf.chapter_title("8. Tabel Hasil Clustering")
    cols_show = ["prov", "kab_kota", "Cluster"]
    
    if data_for_clustering is not None:
        for col in data_for_clustering.columns:
            if col not in cols_show:
                cols_show.append(col)
                
    if "Point Type" in hasil_data.columns:
        cols_show.append("Point Type")
    
    cols_exist = [col for col in cols_show if col in hasil_data.columns]
    pdf.add_dataframe_to_pdf(hasil_data, max_rows=515, cols_to_show=cols_exist)

    # Generate PDF output
    try:
        pdf_output = pdf.output(dest="S")
        
        if isinstance(pdf_output, bytes):
            return pdf_output
        elif isinstance(pdf_output, str):
            return pdf_output.encode("latin-1")
        elif isinstance(pdf_output, bytearray):
            return bytes(pdf_output)
        else:
            st.error(f"Tipe output PDF tidak dikenali: {type(pdf_output)}")
            return None
            
    except Exception as e:
        st.error(f"Gagal menghasilkan output PDF: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None