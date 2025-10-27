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
if 'selected_page_index' not in st.session_state:
    st.session_state['selected_page_index'] = 0

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
    st.session_state['selected_page_index'] = 0
elif selected == "Dataset":
    st.session_state['selected_page_index'] = 1
elif selected == "Clustering":
    st.session_state['selected_page_index'] = 2

if st.session_state['selected_page_index'] == 0:
    render_home_page()
elif st.session_state['selected_page_index'] == 1:
    render_dataset_page()
elif st.session_state['selected_page_index'] == 2:
    render_clustering_page()