import streamlit as st
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io
import time

from modules.plot import (
    render_static_map_to_buffer,
    render_silhouette_plot_to_buffer,
    render_boxplot_to_buffer
)

class PDF(FPDF):
    """Kelas FPDF kustom untuk header dan footer."""
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.multi_cell(0, 5, 'Laporan Hasil Clustering - Sistem Pemetaan Wilayah Berdasarkan Tingkat Pengangguran Terbuka', 0, 'C')
        self.multi_cell(0, 5, 'dan Tingkat Partisipasi Angkatan Kerja di Indonesia dengan K-Means dan DBSCAN', 0, 'C')
        self.ln(5)
        self.set_font('Arial', 'I', 9)
        self.cell(0, 5, f'Tanggal Pembuatan: {time.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_dataframe_to_pdf(self, df, max_rows=100, cols_to_show=None):
        """Menambahkan DataFrame ke PDF sebagai tabel (versi sederhana)."""
        if df is None or df.empty:
            self.chapter_body("Data tabel tidak tersedia.")
            return

        # Pilih kolom yang akan ditampilkan
        if cols_to_show:
            # Jika daftar kolom diberikan, gunakan itu (filter yg ada di df)
            df_display = df[[col for col in cols_to_show if col in df.columns]].copy()
        else:
            # Jika tidak ditentukan, ambil kolom default + SEMUA kolom data
            default_cols = ['ID', 'prov', 'kab_kota', 'Cluster']
            if 'Point Type' in df.columns: default_cols.append('Point Type')
            # Ambil SEMUA kolom yang formatnya mirip data (misal: ada '_')
            data_cols = [col for col in df.columns if col not in default_cols and '_' in col]
            # Gabungkan: default dulu, baru semua kolom data
            cols_to_show_auto = default_cols + sorted(data_cols) # Urutkan kolom data berdasarkan nama
            # Filter lagi untuk memastikan semua kolom ada di df asli
            df_display = df[[col for col in cols_to_show_auto if col in df.columns]].copy()
        if df_display.empty:
            self.chapter_body("Tidak ada kolom yang valid untuk ditampilkan dalam tabel.")
            return

        header_font_size = 7
        data_font_size = 6
        row_height = 5

        self.set_font('Arial', 'B', header_font_size)
        page_width = self.w - 2 * self.l_margin
        num_cols = len(df_display.columns)
        col_width = page_width / num_cols if num_cols > 0 else 10

        # --- Render Header ---
        for col_name in df_display.columns:
            self.cell(col_width, row_height + 1, col_name, border=1, align='C')
        self.ln(row_height + 1)

        # --- Render Data ---
        self.set_font('Arial', '', data_font_size)
        for index, row in df_display.head(max_rows).iterrows():
            y_before_row = self.get_y()
            if y_before_row > (self.h - self.b_margin - (row_height * 2)):
                self.add_page()
                self.set_font('Arial', 'B', header_font_size)
                for col_name in df_display.columns: self.cell(col_width, row_height + 1, col_name, border=1, align='C')
                self.ln(row_height + 1)
                self.set_font('Arial', '', data_font_size)

            for col_name in df_display.columns:
                text = str(row[col_name]); max_chars_per_cell = int(col_width / 1.5)
                if len(text) > max_chars_per_cell: text = text[:max_chars_per_cell - 3] + "..."
                self.cell(col_width, row_height, text, border=1, align='L')
            self.ln(row_height)

        if len(df) > max_rows:
            self.set_font('Arial', 'I', 7)
            self.cell(0, 6, f"... (ditampilkan {max_rows} dari {len(df)} baris data lengkap) ...", 0, 1)


def generate_pdf_report():
    """Mengambil data dari session_state dan membuat laporan PDF dalam bentuk bytes."""

    var = st.session_state.get('var', 'N/A')
    tahun_pilihan = st.session_state.get('tahun_pilihan', ('N/A','N/A'))
    metode = st.session_state.get('metode_pilihan', 'N/A')
    params_dict = st.session_state.get('params', {})
    scores = st.session_state.get('scores')
    hasil_data = st.session_state.get('hasil_data')
    data_for_clustering = st.session_state.get('data_for_clustering')
    gdf_hasil = st.session_state.get('gdf_hasil')

    if hasil_data is None:
        st.warning("Data hasil analisis tidak ditemukan. PDF tidak dapat dibuat.")
        return None

    params_str = ", ".join([f"{k}={v}" for k,v in params_dict.items()]) if params_dict else "N/A"

    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Bagian Input & Metrik ---
    pdf.chapter_title('Parameter Analisis')
    input_text = (f"Variabel Digunakan: {var}\nRentang Tahun: {tahun_pilihan[0]} - {tahun_pilihan[1]}\n"
                f"Metode Clustering: {metode}\nParameter: {params_str}")
    pdf.chapter_body(input_text)
    pdf.chapter_title('Hasil Metrik Evaluasi')
    if scores:
        sil_score_val=scores.get('silhouette'); dbi_score_val=scores.get('dbi'); time_sec_val=scores.get('time_sec')
        sil_str = f"{sil_score_val:.3f}" if isinstance(sil_score_val,(int,float)) else "N/A"
        dbi_str = f"{dbi_score_val:.3f}" if isinstance(dbi_score_val,(int,float)) else "N/A"
        time_str = f"{time_sec_val:.2f} detik" if isinstance(time_sec_val,(int,float)) else "N/A"
        metrics_text=(f"Silhouette Score: {sil_str}\nDavies-Bouldin Index: {dbi_str}\nWaktu Komputasi: {time_str}")
        pdf.chapter_body(metrics_text)
    else: pdf.chapter_body("Skor metrik tidak tersedia.")

    # --- Bagian Plot ---
    pdf.chapter_title('Visualisasi Hasil')

    # --- PETA STATIS ---
    pdf.set_font('Arial', 'I', 10); pdf.cell(0, 10, "Peta Persebaran Cluster (Statis)", 0, 1)
    y_before_map = pdf.get_y()
    if gdf_hasil is not None:
        try:
            map_buffer = render_static_map_to_buffer(gdf_hasil)
            if map_buffer:
                available_width = pdf.w - pdf.l_margin - pdf.r_margin; img_height = available_width * (8/15)
                pdf.image(map_buffer, x=pdf.l_margin, y=y_before_map, w=available_width, h=img_height)
                pdf.set_y(y_before_map + img_height + 5)
            else: pdf.set_font('Arial', '', 10); pdf.multi_cell(0, 5, "(Gagal render gambar peta statis)"); pdf.ln()
        except Exception as e: pdf.set_font('Arial', '', 10); pdf.multi_cell(0, 5, f"Error saat membuat peta statis PDF: {e}"); pdf.ln()
    else: pdf.set_font('Arial', '', 10); pdf.multi_cell(0, 5, "(Data peta tidak tersedia untuk PDF)"); pdf.ln()

    # Cek halaman baru
    if pdf.get_y() > 190 : pdf.add_page()
    y_before_plots = pdf.get_y()

    # --- Silhouette Plot ---
    pdf.set_font('Arial', 'I', 10); pdf.cell(0, 10, "Silhouette Plot", 0, 1)
    x_plot1 = pdf.l_margin; max_y_plot1 = y_before_plots
    if data_for_clustering is not None and scores is not None and scores.get('silhouette') is not None:
        try:
            sil_buffer = render_silhouette_plot_to_buffer(data_for_clustering, hasil_data, scores)
            if sil_buffer:
                plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2; plot_height = plot_width * (5/7)
                pdf.image(sil_buffer, x=x_plot1, y=y_before_plots, w=plot_width, h=plot_height)
                x_after_plot1 = x_plot1 + plot_width + 5; max_y_plot1 = y_before_plots + plot_height
            else:
                plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2
                pdf.set_font('Arial', '', 10); pdf.multi_cell(plot_width, 5, "(Gagal render Silhouette Plot)"); max_y_plot1 = pdf.get_y(); x_after_plot1 = x_plot1 + plot_width + 5; pdf.set_xy(x_after_plot1, y_before_plots)
        except Exception as e: # Error image
            plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2
            pdf.set_font('Arial', '', 10); pdf.multi_cell(plot_width, 5, f"Error Silhouette Plot: {e}"); max_y_plot1 = pdf.get_y(); x_after_plot1 = x_plot1 + plot_width + 5; pdf.set_xy(x_after_plot1, y_before_plots)
    else: # Data tidak cukup
        plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2
        pdf.set_font('Arial', '', 10); pdf.multi_cell(plot_width, 5, "(Data tidak cukup untuk Silhouette Plot)"); max_y_plot1 = pdf.get_y(); x_after_plot1 = x_plot1 + plot_width + 5; pdf.set_xy(x_after_plot1, y_before_plots)

    # --- Box Plot ---
    pdf.set_xy(x_after_plot1 if 'x_after_plot1' in locals() else pdf.l_margin, y_before_plots)
    pdf.set_font('Arial', 'I', 10); pdf.cell(0, 10, "Distribusi Nilai per Cluster", 0, 1)
    pdf.set_x(x_after_plot1 if 'x_after_plot1' in locals() else pdf.l_margin)
    max_y_plot2 = y_before_plots
    try:
        box_buffer = render_boxplot_to_buffer(hasil_data, data_for_clustering)
        if box_buffer:
            plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2; img_height = 80
            pdf.image(box_buffer, x=x_plot1, y=y_before_plots, w=plot_width, h=img_height)
            max_y_plot2 = pdf.get_y() + img_height + 5
        else:
            plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2
            pdf.set_font('Arial', '', 10); pdf.multi_cell(plot_width, 5, "(Gagal render Box Plot)"); max_y_plot2 = pdf.get_y()
    except Exception as e:
        plot_width = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2
        pdf.set_font('Arial', '', 10); pdf.multi_cell(plot_width, 5, f"Error Box Plot: {e}"); max_y_plot2 = pdf.get_y()

    pdf.set_y(max(max_y_plot1, max_y_plot2) + 10)

    # --- Bagian Tabel Hasil ---
    pdf.add_page()
    pdf.chapter_title('Tabel Hasil Clustering (Data Asli)')
    cols_pdf = ['ID', 'prov', 'kab_kota', 'Cluster']
    if 'Point Type' in hasil_data.columns: cols_pdf.append('Point Type')
    data_cols_used = data_for_clustering.columns.tolist() if data_for_clustering is not None else []
    cols_pdf.extend(data_cols_used[:2]); cols_pdf = list(dict.fromkeys(cols_pdf))
    cols_pdf_final = [col for col in cols_pdf if col in hasil_data.columns]

    pdf.add_dataframe_to_pdf(hasil_data, max_rows=100, cols_to_show=cols_pdf_final)

    # Output PDF sebagai bytes
    try:
        pdf_output_bytes = pdf.output(dest='S').encode('latin-1')
        return pdf_output_bytes
    except Exception as e:
        st.error(f"Gagal menghasilkan output PDF: {e}")
        return None