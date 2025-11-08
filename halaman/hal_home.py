import streamlit as st
import pandas as pd
import plotly.express as px
from modules.konten import judul, string1, string2, string3, cara_penggunaan, question1, question2, answer1, answer2

def render_home_page():    
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
    st.title(judul)
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
      if st.button("📊 Lihat Dataset", use_container_width=True):
        st.session_state['selected_page_index'] = 1
        st.rerun()
    with col_btn2:
      if st.button("🚀 Mulai Clustering", type="primary", use_container_width=True):
        st.session_state['selected_page_index'] = 2
        st.rerun()
    
    st.write("")

    # Manual Book Section
    st.markdown("""
    <div style="background-color:#E0F7FA; padding:20px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px #aaaaaa;">
      <h3>📖 Manual Book</h3>
      <p>Apabila membutuhkan panduan lebih lanjut, klik tombol dibawah.</p>
      <a href="https://drive.google.com/file/d/14LN6MrMFD35S1m-PRDMP186J7Dki0mZ0/view?usp=sharing" target="_blank">
        <button style="background-color:#00796B; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer;">
          Buka Manual Book
        </button>
      </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Cara penggunaan singkat
    st.subheader("CARA PENGGUNAAN SINGKAT")
    st.markdown(cara_penggunaan, unsafe_allow_html=True)
    st.write("")
    
    # Latar Belakang
    st.subheader("LATAR BELAKANG")
    st.markdown(string1, unsafe_allow_html=True)
    st.write("")
    
    # Grafik Tren Nasional
    bikin_grafik_tren_nasional()
        
    st.divider()
    
    # Deskripsi Sistem
    st.write("")
    st.markdown(string2, unsafe_allow_html=True)
    st.write("")
    st.markdown(string3, unsafe_allow_html=True)
    st.write("")
    
    st.subheader("FAQ")
    st.expander(question1, expanded=False).markdown(answer1, unsafe_allow_html=True)
    st.expander(question2, expanded=False).markdown(answer2, unsafe_allow_html=True)

def bikin_grafik_tren_nasional():
    st.subheader("Tren Rata-Rata TPT dan TPAK Nasional (2018-2024)")
    
    # Data nasional
    years = list(range(2018, 2025))
    tpt_nasional = [5.30, 5.23, 7.07, 6.49, 5.86, 5.32, 4.91]
    tpak_nasional = [67.31, 67.53, 67.77, 67.80, 68.63, 69.48, 70.63]
    
    if len(years) == len(tpt_nasional) == len(tpak_nasional):
      df_trend_nasional = pd.DataFrame({
        'Tahun': years,
        'TPT Nasional (%)': tpt_nasional,
        'TPAK Nasional (%)': tpak_nasional
      })
      
      col_trend1, col_trend2 = st.columns(2)

      with col_trend1:
        # Grafik TPT
        fig_tpt = px.line(
          df_trend_nasional,
          x='Tahun',
          y='TPT Nasional (%)',
          title="Tren TPT Nasional",
          markers=True,
          labels={'Tahun': '', 'TPT Nasional (%)': 'TPT (%)'} # Sumbu X dikosongkan agar tidak duplikat
        )
        fig_tpt.update_layout(xaxis_tickformat='d')
        st.plotly_chart(fig_tpt, use_container_width=True)

      with col_trend2:
        # Grafik TPAK
        fig_tpak = px.line(
          df_trend_nasional,
          x='Tahun',
          y='TPAK Nasional (%)',
          title="Tren TPAK Nasional",
          markers=True,
          labels={'Tahun': '', 'TPAK Nasional (%)': 'TPAK (%)'}
        )
        fig_tpak.update_layout(xaxis_tickformat='d')
        st.plotly_chart(fig_tpak, use_container_width=True)

    else:
      st.warning("Jumlah data TPT/TPAK tidak cocok dengan rentang tahun 2018-2024. Grafik tren tidak dapat dibuat.")
      st.write(f"Years: {len(years)}, TPT: {len(tpt_nasional)}, TPAK: {len(tpak_nasional)}") # Info debug
    return