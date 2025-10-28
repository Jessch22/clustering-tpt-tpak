from streamlit_option_menu import option_menu
import streamlit as st
from halaman.hal_home import render_home_page
from halaman.hal_dataset import render_dataset_page
from halaman.hal_clustering import render_clustering_page

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="Pemetaan Wilayah TPT dan TPAK",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- SESSION DEFAULT ---
if "selected_page_index" not in st.session_state:
    st.session_state["selected_page_index"] = 0

# --- MENU LIST ---
menu_list = ["Home", "Dataset", "Clustering"]

# --- OPTION MENU ---
selected = option_menu(
    menu_title=None,
    options=menu_list,
    icons=["house", "database", "gear"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#192734"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {
            "font-size": "18px",
            "color": "white",
            "text-align": "center",
            "margin": "0px",
        },
        "nav-link-selected": {"background-color": "#215C3B", "color": "white"},
    },
    default_index=st.session_state["selected_page_index"],
)

# Jika user klik tab menu, update state dan rerun
current_index = menu_list.index(selected)
if current_index != st.session_state["selected_page_index"]:
    st.session_state["selected_page_index"] = current_index
    st.rerun()

# --- ROUTING ---
page_index = st.session_state["selected_page_index"]

if page_index == 0:
    render_home_page()
elif page_index == 1:
    render_dataset_page()
elif page_index == 2:
    render_clustering_page()
