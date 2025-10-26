import streamlit as st
from modules.konten import penjelasan_tpt, penjelasan_tpak
from utils.session import opsi_tahun
from streamlit_folium import st_folium
import time

from modules.analysis import run_analysis
from modules.plot import render_metrics_and_silhouette, render_boxplot, render_scatter_plots # Hapus render_static_map jika tidak dipakai lagi
from utils.saveaspdf import generate_pdf_report
from utils.session import opsi_tahun

def render_clustering_page():
  st.title("Sistem Pemetaan Wilayah Berdasarkan Tingkat Pengangguran Terbuka dan Tingkat Partisipasi Angkatan Kerja di Indonesia dengan K-Means dan DBSCAN")
  path = "DATASET.xlsx"
  sheet = "Populasi"

  col1, col2 = st.columns([0.5, 0.5])
  with col1:
    #* Variabel & Tahun
    st.subheader("PILIH VARIABEL & TAHUN")

    tpt = st.checkbox(
      "Tingkat Pengangguran Terbuka (TPT)",
      key="tpt_checked",
      value=st.session_state.get("tpt_checked", False)
    )
    tpak = st.checkbox(
      "Tingkat Partisipasi Angkatan Kerja (TPAK)",
      key="tpak_checked",
      value=st.session_state.get("tpak_checked", False)
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
      value=tuple(st.session_state.get('tahun_pilihan', (opsi_tahun[0], opsi_tahun[-1]))),
      key="tahun_pilihan",
      disabled=not enable_year_picker
    )

    # Normalisasi tipe tahun di session_state
    tp_after = st.session_state.get('tahun_pilihan')
    if isinstance(tp_after, str):
      st.session_state['tahun_pilihan'] = (tp_after, tp_after)
    elif isinstance(tp_after, list):
      st.session_state['tahun_pilihan'] = tuple(tp_after)
    st.divider()

    #* Metode & Parameter
    st.subheader("METODE & PARAMETER")
    metode_options = ["K-Means", "DBSCAN"]
    metode = st.radio(
      "Pilih Metode", metode_options,
      key="metode_pilihan",
      index=metode_options.index(st.session_state.get("metode_pilihan", metode_options[0])),
      label_visibility="collapsed"
    )
    params = {}
    if metode == "K-Means":
      st.markdown("K (Jumlah cluster yang ingin dibentuk)")
      k_value = st.slider("K", 2, 6, key="k_value", value=st.session_state.get("k_value", 2), label_visibility="collapsed")
      params['k'] = k_value
    elif metode == "DBSCAN":
      st.markdown("Epsilon (Jarak maksimum antar titik untuk dianggap sebagai tetangga)")
      eps_value = st.slider("Epsilon", 0.1, 25.0, step=0.1, key="eps_value", value=st.session_state.get("eps_value", 0.1), label_visibility="collapsed")
      st.markdown("MinPts (Jumlah minimum titik dalam radius Epsilon untuk membentuk sebuah cluster)")
      minpts_value = st.slider("MinPts", 2, 42, step=1, key="minpts_value", value=st.session_state.get("minpts_value", 2), label_visibility="collapsed")
      params['eps'] = eps_value
      params['minpts'] = minpts_value

    st.divider()
    run_button = st.button("Jalankan Clustering & Peta", type="primary", use_container_width=True)
    if run_button:
      if not var:
        st.error("Silakan pilih minimal satu variabel (TPT atau TPAK) terlebih dahulu sebelum menjalankan.")
      else:
        with st.spinner("Memproses data dan membuat peta..."):
          run_analysis(
            var,
            st.session_state['tahun_pilihan'],
            st.session_state['metode_pilihan'],
            params,
            path,
            sheet
          )
          
  # BAGIAN 3 - OUTPUT HASIL
  with col2: # Metrik & Silhouette
      st.subheader("HASIL METRIK & GRAFIK")
      render_metrics_and_silhouette(
          st.session_state.get('scores'),
          st.session_state.get('hasil_data'),
          st.session_state.get('data_for_clustering')
      )
      
  st.divider(); st.subheader("DISTRIBUSI CLUSTER (BOX PLOT)") # Box Plot
  render_boxplot(
      st.session_state.get('hasil_data'),
      st.session_state.get('data_for_clustering')
  )
  
  st.divider(); st.subheader("PETA WILAYAH (INTERAKTIF)") # Peta
  if st.session_state.get('map_object') is not None:
      with st.spinner("Memuat peta..."):
          try:
              st_folium(st.session_state['map_object'], use_container_width=True, height=600, returned_objects=[])
          except Exception as e:
              st.error(f"Gagal menampilkan peta: {e}"); st.container(height=600)
  else: st.container(height=600)

  st.divider() # Scatter Plot
  render_scatter_plots(
      st.session_state.get('hasil_data'),
      st.session_state.get('data_for_clustering')
  )

  st.divider(); st.subheader("TABEL HASIL") # Tabel & Tombol
  if st.session_state.get('hasil_data') is not None:
      st.dataframe(st.session_state['hasil_data'], use_container_width=True)
      col_dl_1, col_dl_2 = st.columns([1,1])
      with col_dl_1: # Tombol Download
        pdf_bytes = None; generate_pdf = False
        if st.session_state.get('hasil_data') is not None:
          generate_pdf = True
        try: pdf_bytes = generate_pdf_report()
        except Exception as e_pdf: st.error(f"Gagal mempersiapkan data PDF: {e_pdf}"); pdf_bytes = None
        if generate_pdf and pdf_bytes:
          st.download_button(
            label="📄 Download PDF Report", data=pdf_bytes,
            file_name=f"clustering_report_{st.session_state.get('metode_pilihan','hasil')}_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", type="primary", use_container_width=True)
        else:
          st.button("📄 Download PDF Report", disabled=True, help="Jalankan analisis...", use_container_width=True)
      with col_dl_2: # Tombol Clear
          if st.button("❌ Clear Current Result", use_container_width=True):
            keys_to_clear = ['hasil_data', 'scores', 'data_for_clustering', 'gdf_hasil', 'map_object', 'var', 'params']
            for key in keys_to_clear:
              if key in st.session_state: st.session_state[key] = None
            st.success("Hasil saat ini dihapus.")
            st.rerun()
  else: st.info("Belum ada hasil clustering...")