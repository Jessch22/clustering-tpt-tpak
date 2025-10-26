import streamlit as st
from streamlit_option_menu import option_menu

from halaman.hal_home import render_home_page
from halaman.hal_dataset import render_dataset_page
from halaman.hal_clustering import render_clustering_page

#* KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title= "Pemetaan Wilayah TPT dan TPAK",
    page_icon= ":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#* INISIALISASI APLIKASI ==========
# initialize_session_state()

#* NAVBAR UTAMA ==========
selected = option_menu(
    menu_title=None,
    # Opsi dan iconnya
    options=["Home", "Dataset", "Clustering"],
    icons=["house",  "database", "gear"],
    
    orientation="horizontal",
    styles={
        "container": {
          "padding": "0!important",
          "background-color": "#192734"
        },
        "icon": {
          "color": "white", 
          "font-size": "18px"
        },
        "nav-link": {
            "font-size": "18px",
            "color": "white",
            "text-align": "center",
            "margin": "0px"
          },
        "nav-link-selected": {"background-color": "#215C3B", "color": "white"},
    }
)

#* ROUTING HALAMAN ==========
if selected == "Home":
    render_home_page()
elif selected == "Dataset":
    render_dataset_page()
elif selected == "Clustering":
    render_clustering_page()
