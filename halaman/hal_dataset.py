import streamlit as st
import pandas as pd
from modules.data_processing import muat_data, replace_non_numeric
from modules.konten import penjelasan_dataset

def render_dataset_page():
    with st.spinner("Memuat dataset..."):
      st.title("TAMPILAN DATASET ASLI")
      st.markdown("Dataset bersumber dari BPS (Badan Pusat Statistik) per provinsi.")

      path = "DATASET.xlsx"
      sheet = "Populasi"
      
      try:
          # 1. Muat data
          df_raw = muat_data(path, sheet)

          if df_raw.empty:
            # Debug error muat data
            st.error("Gagal memuat data atau data kosong.")
          else:
            df_display = replace_non_numeric(df_raw)
            st.dataframe(df_display, use_container_width=True)
            st.caption(f"Total {df_display.shape[0]} baris dan {df_display.shape[1]} kolom.")

      except FileNotFoundError:
        # Debug path file tidak ditemukan
        st.error(f"File '{path}' tidak ditemukan.")
      except Exception as e:
        # Debug error umum
        st.error(f"Terjadi kesalahan saat memuat atau menampilkan data: {e}")
        
      with st.expander("Penjelasan Kolom Dataset"):
          st.markdown(penjelasan_dataset, unsafe_allow_html=True)