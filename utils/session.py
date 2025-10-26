import streamlit as st

# Inisialisasi daftar tahun
opsi_tahun = [str(y) for y in range(2018, 2025)]

def get_var():
    return st.session_state.get('var', '')

def get_tahun():
    return st.session_state.get('tahun_pilihan', (opsi_tahun[0], opsi_tahun[-1]))

def get_params():
    return st.session_state.get('params', {})

def get_metode():
    return st.session_state.get('metode_pilihan', 'K-Means')

def get_tpt_checked():
    return st.session_state.get('tpt_checked', False)

def get_tpak_checked():
    return st.session_state.get('tpak_checked', False)

def reset_clustering_state():
    for key in ['var', 'tahun_pilihan', 'tpt_checked', 'tpak_checked', 'metode_pilihan', 'params', 'hasil_data', 'scores', 'gdf_hasil', 'data_for_clustering', 'map_object', 'cluster_color_map']:
        if key in st.session_state:
            st.session_state[key] = None