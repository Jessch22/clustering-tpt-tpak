import streamlit as st
from modules.konten import penjelasan_tpt, penjelasan_tpak
from utils.session import opsi_tahun
from streamlit_folium import st_folium
import time
import pandas as pd
import numpy as np

from modules.analysis import run_analysis
from modules.plot import (
    render_kmeans_helpers,
    render_metrics_and_silhouette,
    render_boxplot,
    render_scatter_plots,
    render_dbscan_helpers,
)
from utils.saveaspdf import generate_pdf_report


def render_clustering_page():
    st.markdown("""
    <style>
    h1 {
        text-align: center;
    }
    
    @media (max-width: 600px) {
        h1 {
            font-size: 1.6rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    st.title(
        "Sistem Pemetaan Wilayah Berdasarkan Tingkat Pengangguran Terbuka dan Tingkat Partisipasi Angkatan Kerja di Indonesia dengan K-Means dan DBSCAN"
    )
    path = "DATASET.xlsx"
    sheet = "Populasi"

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        # * Variabel & Tahun
        st.subheader("PILIH VARIABEL & TAHUN")

        tpt = st.checkbox(
            "Tingkat Pengangguran Terbuka (TPT)",
            key="tpt_checked",
            value=st.session_state.get("tpt_checked", False),
        )
        tpak = st.checkbox(
            "Tingkat Partisipasi Angkatan Kerja (TPAK)",
            key="tpak_checked",
            value=st.session_state.get("tpak_checked", False),
        )

        if tpt and tpak:
            var = "tpt_tpak"
        elif tpt:
            var = "tpt"
        elif tpak:
            var = "tpak"
        else:
            var = ""

        enable_year_picker = bool(var)
        if not enable_year_picker:
            st.info("Pilih variabel (TPT/TPAK) terlebih dahulu")

        tahun_pilihan = st.select_slider(
            "Pilih Range Tahun",
            options=opsi_tahun,
            value=tuple(
                st.session_state.get(
                    "tahun_pilihan", (opsi_tahun[0], opsi_tahun[-1])
                )
            ),
            key="tahun_pilihan",
            disabled=not enable_year_picker,
        )

        # Normalisasi tipe tahun di session_state
        tp_after = st.session_state.get("tahun_pilihan")
        if isinstance(tp_after, str):
            st.session_state["tahun_pilihan"] = (tp_after, tp_after)
        elif isinstance(tp_after, list):
            st.session_state["tahun_pilihan"] = tuple(tp_after)
        
        if var:
            n_vars = (1 if tpt else 0) + (1 if tpak else 0)
            
            current_tahun_range = st.session_state["tahun_pilihan"] 
            start_year = int(current_tahun_range[0])
            end_year = int(current_tahun_range[1])
            n_years = (end_year - start_year) + 1
            
            # Dimensi
            D = n_vars * n_years
            
            selected_vars_list = []
            if tpt: selected_vars_list.append("TPT")
            if tpak: selected_vars_list.append("TPAK")
            
            selected_years_list = [str(y) for y in range(start_year, end_year + 1)]

            feature_list = []
            for v_name in selected_vars_list:
                for y_name in selected_years_list:
                    feature_list.append(f"{v_name}_{y_name}")
            
            MAX_FEATURES_TO_SHOW = 7
            if D > 0:
                if D <= MAX_FEATURES_TO_SHOW:
                    feature_str = ", ".join(feature_list)
                    st.info(f"**Dimensi (D) = {D}** ({feature_str})")
                else:
                    feature_str_short = f"{feature_list[0]}, {feature_list[1]}, {feature_list[2]} ... {feature_list[-2]}, {feature_list[-1]}"
                    st.info(f"**Dimensi (D) = {D}** (Fitur: {feature_str_short}, dst.)")
            else:
                st.info(f"**Dimensi (D) = 0**")
                
        st.divider()
        
        # * Metode & Parameter
        st.subheader("METODE & PARAMETER")
        metode_options = ["K-Means", "DBSCAN"]
        metode = st.radio(
            "Pilih Metode",
            metode_options,
            key="metode_pilihan",
            index=metode_options.index(
                st.session_state.get("metode_pilihan", metode_options[0])
            ),
            label_visibility="collapsed",
        )
        params = {}
        if metode == "K-Means":
            optimal_k = st.checkbox(
                "Nilai K Optimal",
                key="optimal_k",
                value=st.session_state.get("optimal_k", False),
                help="Jika dicentang, sistem secara otomatis memilih K dengan Silhouette Score terbaik."
            )
                        
            st.markdown("K (Jumlah cluster yang ingin dibentuk)")
            k_value = st.slider(
                "K",
                2,
                6,
                key="k_value",
                value=st.session_state.get("k_value", 2),
                label_visibility="collapsed",
                disabled=optimal_k,
            )
            params["k"] = k_value
            params["optimal_k"] = optimal_k
            
        elif metode == "DBSCAN":
            optimal_dbscan = st.checkbox(
                "Nilai Epsilon & MinPts Optimal", 
                key="optimal_dbscan", 
                value=st.session_state.get("optimal_dbscan", False),
                help="""
                Jika dicentang, sistem akan otomatis:
                - PCA digunakan jika Dimensi≥3
                - MinPts diatur sesuai Dimensi+1
                - Mencari nilai epsilon terbaik dengan K-distance Knee.
                """
            )
            
            st.markdown("Epsilon (Jarak maksimum antar titik untuk dianggap sebagai tetangga)")
            eps_value = st.slider(
                "Epsilon",
                0.1,
                25.0,
                step=0.1,
                key="eps_value",
                value=st.session_state.get("eps_value", 0.5),
                label_visibility="collapsed",
                disabled=optimal_dbscan,
            )
            st.markdown("MinPts (Jumlah minimum titik dalam radius Epsilon untuk membentuk sebuah cluster)")
            minpts_value = st.slider(
                "MinPts",
                2,
                20,
                step=1,
                key="minpts_value",
                value=st.session_state.get("minpts_value", 3),
                label_visibility="collapsed",
                disabled=optimal_dbscan,
            )
            
            use_pca_manual = st.checkbox(
                "Terapkan PCA (jika D >= 3)",
                key="use_pca_manual",
                value=st.session_state.get("use_pca_manual", True),
                disabled=optimal_dbscan,
                help="""
                - Wajib untuk Dimensi yang lebih dari sama dengan 3 (D>=3).
                - Tujuan: Mengurangi Dimensi agar DBSCAN lebih efektif dalam menemukan cluster.
                - Jika tidak dicentang, DBSCAN akan dijalankan pada Dimensi asli.
                """
            )
            
            params["eps"] = eps_value
            params["minpts"] = minpts_value
            params["optimal_dbscan"] = optimal_dbscan
            params["use_pca_manual"] = use_pca_manual

        st.divider()
        run_button = st.button(
            "Jalankan Clustering", type="primary", use_container_width=True
        )
        if run_button:
            if not var:
                st.error(
                    "Silakan pilih minimal satu variabel (TPT atau TPAK) terlebih dahulu sebelum menjalankan."
                )
            else:
                logger = st.sidebar.status("Memulai proses clustering...", expanded=True)
                with st.spinner("Memproses data dan membuat peta..."):
                    run_analysis(
                        var,
                        st.session_state["tahun_pilihan"],
                        st.session_state["metode_pilihan"],
                        params,
                        path,
                        sheet,
                        logger
                    )
                    
                    logger.update(label="Proses Selesai!", state="complete", expanded=True)

    # BAGIAN 3 - OUTPUT HASIL
    with col2:  # Metrik & Silhouette
        st.subheader("HASIL METRIK & GRAFIK")
        render_metrics_and_silhouette(
            st.session_state.get("scores"),
            st.session_state.get("hasil_data"),
            st.session_state.get("data_for_clustering"),
        )

    metode_terpilih = st.session_state.get("metode_pilihan", "K-Means")
    if metode_terpilih == "DBSCAN":
        render_dbscan_helpers(
            st.session_state.get("dbscan_elbow_data"),
            st.session_state.get("dbscan_elbow_minpts"),
            st.session_state.get("dbscan_minpts_plot_data"),
            st.session_state.get("dbscan_elbow_knee")
        )
        st.divider()
        
    elif metode_terpilih == "K-Means":
        render_kmeans_helpers(
            st.session_state.get("kmeans_k_search_data")
        )
        # Tambahkan divider hanya jika tabelnya benar-benar ditampilkan
        if st.session_state.get("kmeans_k_search_data") is not None:
            st.divider()
    st.subheader("JUMLAH WILAYAH PER CLUSTER", help="Tabel ini menunjukkan jumlah wilayah yang termasuk dalam setiap cluster hasil clustering.")
    if st.session_state.get("hasil_data") is not None:
        hasil_df = st.session_state["hasil_data"]
        if "Cluster" in hasil_df.columns:
            cluster_counts = hasil_df["Cluster"].value_counts().sort_index()
            counts_df = pd.DataFrame({
                "Cluster": cluster_counts.index.astype(str),
                "Jumlah_wilayah": cluster_counts.values
            })
            counts_df = counts_df.reset_index(drop=True)
            st.table(counts_df)
        else:
            st.warning("Kolom 'Cluster' tidak ditemukan pada hasil clustering.")
    else:
        st.warning("Belum ada hasil clustering untuk menampilkan jumlah wilayah per cluster.")
        
    st.divider()
    
    st.subheader("DISTRIBUSI CLUSTER (BOX PLOT)", help="Box plot digunakan untuk memahami distribusi nilai variabel pada tiap cluster yang terbentuk. Perhatikan median, kuartil, dan outlier pada setiap box plot untuk melihat karakteristik cluster.")
    render_boxplot(
        st.session_state.get("hasil_data"),
        st.session_state.get("data_for_clustering"),
    )

    st.divider()
    st.subheader("PETA WILAYAH (INTERAKTIF)", help="Peta interaktif yang menunjukkan sebaran geografis berdasarkan cluster yang terbentuk.")
    if st.session_state.get("map_object") is not None:
        with st.spinner("Memuat peta..."):
            try:
                st_folium(
                    st.session_state["map_object"],
                    use_container_width=True,
                    height=600,
                    returned_objects=[],
                )
            except Exception as e:
                st.error(f"Gagal menampilkan peta: {e}")
                st.container(height=600)
    else:
        st.container(height=600)

    st.divider()
    st.subheader(
        "ANALISIS PEMISAHAN CLUSTER", 
        help="Visualisasi ini digunakan untuk menganalisis pemisahan cluster secara visual.")
    st.caption("""
    **Cara Kerja:** Pilih satu 'Variabel Fokus' untuk melihat plot distribusinya (KDE) dan membandingkan sebarannya (Scatter Plot) terhadap semua variabel lain.
    
    **Cara Membaca:**
    - **Plot Distribusi (KDE):** Menunjukkan tumpang-tindih cluster pada *satu* variabel.
    - **Plot Hubungan (Scatter):** Menunjukkan seberapa baik *dua* variabel memisahkan gumpalan cluster.
    """)
    render_scatter_plots(
        st.session_state.get("hasil_data"),
        st.session_state.get("data_for_clustering"),
    )

    st.divider()
    st.subheader("TABEL & LAPORAN HASIL")

    if st.session_state.get("hasil_data") is not None:
        df_hasil = st.session_state["hasil_data"]
        df_cluster_data = st.session_state.get("data_for_clustering")
        
        # 1. Tentukan kolom dasar yang selalu ingin ditampilkan
        cols_to_show = ['prov', 'kab_kota', 'Cluster']
        
        # 2. Tambahkan 'Point Type' jika ada (hasil DBSCAN)
        if 'Point Type' in df_hasil.columns:
            cols_to_show.append('Point Type')
            
        # 3. Ambil kolom variabel yang DIPILIH (dari data_for_clustering)
        if df_cluster_data is not None and not df_cluster_data.empty:
            selected_var_cols = df_cluster_data.columns.tolist()
            # Tambahkan kolom variabel yang dipilih ke daftar
            for col in selected_var_cols:
                if col not in cols_to_show:
                    cols_to_show.append(col)
        
        # 4. Filter dataframe utama (df_hasil) agar hanya menampilkan kolom yang ada di cols_to_show
        final_cols_exist = [col for col in cols_to_show if col in df_hasil.columns]
        df_display = df_hasil[final_cols_exist]
        
        # 5. Tampilkan dataframe yang sudah difilter
        st.dataframe(df_display, use_container_width=True)

        col_dl_1, col_dl_2 = st.columns([1, 1])
        
        with col_dl_1:
            if st.button("📄 Simpan Laporan PDF", type="primary", use_container_width=True):
                try:
                    var_info = {
                        "TPT": st.session_state.get("tpt_checked", False),
                        "TPAK": st.session_state.get("tpak_checked", False),
                    }
                    metode = st.session_state.get("metode_pilihan", "K-Means")
                    params = (
                        {"k": st.session_state.get("k_value", 2)}
                        if metode == "K-Means"
                        else {
                            "eps": st.session_state.get("eps_value", None),
                            "minpts": st.session_state.get("minpts_value", None),
                        }
                    )

                    with st.spinner("Menyusun laporan PDF..."):
                        pdf_bytes = generate_pdf_report()


                        if pdf_bytes:
                            st.download_button(
                                label="💾 Unduh Laporan PDF",
                                data=pdf_bytes,
                                file_name=f"laporan_clustering_{metode}_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True,
                            )
                        else:
                            st.error("Gagal membuat laporan PDF.")
                except Exception as e:
                    st.error(f"Gagal membuat laporan PDF: {e}")

        with col_dl_2:
            if st.button("❌ Hapus Hasil Sekarang", use_container_width=True):
                # HANYA HAPUS KEY HASIL, BUKAN KEY WIDGET
                keys_to_clear = [
                    # Data Hasil Utama
                    "hasil_data",
                    "scores",
                    "data_for_clustering",
                    "gdf_hasil",
                    "map_object",
                    "var",
                    "params",
                    
                    "dbscan_elbow_data",
                    "dbscan_elbow_minpts",
                    "dbscan_minpts_plot_data",
                    "dbscan_elbow_knee",
                    "kmeans_k_search_data"
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        st.session_state[key] = None
                
                st.success("Hasil analisis saat ini telah dihapus.")
                st.rerun()
    else:
        st.info("Belum ada hasil clustering...")