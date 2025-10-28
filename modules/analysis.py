import streamlit as st
import pandas as pd
import geopandas as gpd
import time
import os
from modules.data_processing import muat_data, preprocessing_data, kmeans_clustering, dbscan_clustering
from modules.plot import create_folium_map

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import re
from typing import Optional


def run_analysis(var, tahun_pilihan, metode_terpilih, params, path, sheet):
    """Fungsi untuk Menjalankan analisis clustering dari logic/computation.py"""

    # Initialize/clear relevant session state keys
    st.session_state['hasil_data'] = None
    st.session_state['scores'] = None
    st.session_state['gdf_hasil'] = None
    st.session_state['data_for_clustering'] = None
    st.session_state['map_object'] = None
    st.session_state['cluster_color_map'] = None


     # *** TAMBAHAN: SIMPAN var DAN params KE SESSION_STATE ***
    st.session_state['var'] = var
    st.session_state['params'] = params

    start_proc = time.perf_counter()

    try:
        # ============ 1. Muat dan Preprocessing Data ============
        df_raw = muat_data(path, sheet)
        if df_raw.empty:
            st.error("Dataset kosong. Pastikan file DATASET.xlsx benar dan memiliki sheet yang dipilih.")
            return

        # Pastikan ada kolom ID, prov, kab_kota
        required_cols = ['ID', 'prov', 'kab_kota']
        for c in required_cols:
            if c not in df_raw.columns:
                st.error(f"Kolom '{c}' tidak ditemukan di dataset. Pastikan dataset memiliki kolom ID, prov, kab_kota.")
                return

        identity_cols = df_raw[['ID', 'prov', 'kab_kota']].copy()

        # Preprocessing: menghasilkan data_clean, data_norm, data_splits, dsb.
        datacol_num, data_replace, data_clean, data_norm, data_splits = preprocessing_data(df_raw)

        # ============ 2. Pilih data untuk clustering ============
        start_year = tahun_pilihan[0]
        end_year = tahun_pilihan[1]
        nama_data = f"{var}_{start_year}" if start_year == end_year else f"{var}_{start_year}_{end_year}"

        if nama_data not in data_splits:
            st.error(f"Data split '{nama_data}' tidak ditemukan.")
            return

        data_for_clustering = data_splits[nama_data]
        st.session_state['data_for_clustering'] = data_for_clustering

        # ================ 3. Lakukan Clustering =================
        labels = None
        point_type = None
        sil_score = None
        dbi_score = None

        with st.spinner("Menjalankan clustering..."):
            if metode_terpilih == "K-Means":
                k = params.get('k', 3)
                hasil_cluster = kmeans_clustering(data_for_clustering, k)
                labels = hasil_cluster.get('labels')
                sil_score = hasil_cluster.get('silhouette', None)
                dbi_score = hasil_cluster.get('dbi', None)
                st.success(f"K-Means selesai. Jumlah label: {len(labels) if labels is not None else 'None'}")

            elif metode_terpilih == "DBSCAN":
                eps = params.get('eps', 0.5)
                minpts = params.get('minpts', 5)
                hasil_cluster = dbscan_clustering(data_for_clustering, eps, minpts)
                labels = hasil_cluster.get('labels')
                point_type = hasil_cluster.get('point_type', None)
                sil_score = hasil_cluster.get('silhouette', None)
                dbi_score = hasil_cluster.get('dbi', None)
                st.success(f"DBSCAN selesai. Jumlah label: {len(labels) if labels is not None else 'None'}")

            else:
                st.error("Metode clustering tidak dikenali.")
                return

        # ============== 4. Buat Tabel Hasil Akhir ===============
        with st.spinner("Membuat visualisasi..."):
            if labels is not None and data_clean.shape[0] == len(labels):
                df_output_temp = pd.concat([identity_cols.reset_index(drop=True), data_clean.reset_index(drop=True)], axis=1)
                df_output_temp['Cluster'] = labels

                if metode_terpilih == "DBSCAN":
                    if point_type is not None and len(point_type) == data_clean.shape[0]:
                        df_output_temp['Point Type'] = point_type
                    else:
                        st.warning("Gagal menambahkan kolom 'Point Type' (DBSCAN). Akan diisi None.")
                        df_output_temp['Point Type'] = None

                st.session_state['hasil_data'] = df_output_temp

                end_proc = time.perf_counter()
                elapsed_sec = end_proc - start_proc
                st.session_state['scores'] = {'silhouette': sil_score, 'dbi': dbi_score, 'time_sec': elapsed_sec}

                st.session_state['map_object'] = None

                try:
                    geojson_path = r'geojson/38 Provinsi Indonesia - Kabupaten.json'
                    if not os.path.exists(geojson_path):
                        raise FileNotFoundError(f"GeoJSON tidak ditemukan. Letakkan file GeoJSON di: {geojson_path}")

                    gdf = gpd.read_file(geojson_path)

                    NAMA_KAB_KOTA = _detect_name_column(gdf)
                    if NAMA_KAB_KOTA is None:
                        st.error("Tidak dapat mendeteksi kolom nama wilayah di GeoJSON. Kolom yang biasa: WADMKK, NAME_2, NAMA, dsb.")
                        st.session_state['map_object'] = None
                        return

                    gdf['orig_name'] = gdf[NAMA_KAB_KOTA].astype(str)

                    gdf['norm_name'] = _normalize_name_series(gdf['orig_name'])
                    gdf['join_name'] = _make_join_key(gdf['norm_name'])

                    PROV_COL = _detect_prov_column(gdf)
                    if PROV_COL:
                        gdf['prov_g'] = _normalize_name_series(gdf[PROV_COL].astype(str))
                    else:
                        gdf['prov_g'] = ""

                    df_hasil_final_map = st.session_state['hasil_data'].copy()

                    # --- Hapus kab/kota yang kosong, NaN, atau string "None" ---
                    df_hasil_final_map = df_hasil_final_map[
                        df_hasil_final_map['kab_kota'].notna() &
                        (df_hasil_final_map['kab_kota'].astype(str).str.lower() != 'none') &
                        (df_hasil_final_map['kab_kota'].astype(str).str.strip() != '')
                    ]

                    # --- Lanjutkan seperti biasa ---
                    df_hasil_final_map['orig_name_df'] = df_hasil_final_map['kab_kota'].astype(str)
                    df_hasil_final_map['norm_name_df'] = _normalize_name_series(df_hasil_final_map['orig_name_df'])
                    df_hasil_final_map['join_name'] = _make_join_key(df_hasil_final_map['norm_name_df'])


                    if 'prov' in df_hasil_final_map.columns:
                        df_hasil_final_map['prov_d'] = _normalize_name_series(df_hasil_final_map['prov'].astype(str))
                    else:
                        df_hasil_final_map['prov_d'] = ""

                    try:
                        gdf_merged = gdf.merge(
                            df_hasil_final_map[['join_name', 'Cluster', 'prov_d', 'orig_name_df']],
                            left_on='join_name',
                            right_on='join_name',
                            how='left',
                            validate="m:1"
                        )

                        unmatched = gdf_merged[gdf_merged['Cluster'].isna()][['orig_name', 'prov']]
                        if not unmatched.empty:
                            st.warning(f"{len(unmatched)} wilayah tidak cocok dengan dataset:")
                            st.dataframe(unmatched)

                    except Exception as e_merge:


                        gdf_merged = gdf.merge(
                            df_hasil_final_map[['join_name', 'Cluster', 'prov_d', 'orig_name_df']],
                            left_on='join_name',
                            right_on='join_name',
                            how='left'
                        )

                    def _choose_prov(row):
                        try:
                            if isinstance(row.get('prov_d', None), str) and row.get('prov_d', "").strip():
                                return row.get('prov_d', "")
                            if isinstance(row.get('prov_g', None), str) and row.get('prov_g', "").strip():
                                return row.get('prov_g', "")
                        except Exception:
                            pass
                        return ""

                    gdf_merged['display_name'] = gdf_merged['orig_name']  # original geojson name
                    gdf_merged['prov'] = gdf_merged.apply(_choose_prov, axis=1)

                    gdf_merged['display_name'] = gdf_merged['display_name'].fillna(gdf_merged.get('orig_name_df', ""))
                    gdf_merged['prov'] = gdf_merged['prov'].fillna("").astype(str)

                    st.session_state['gdf_hasil'] = gdf_merged

                    try:
                        n_matched = gdf_merged['Cluster'].notna().sum()
                        n_total = len(gdf_merged)
                        st.info(f"Wilayah yang berhasil dicocokkan: {n_matched-1}/{n_total-1}")
                    except Exception:
                        pass

                    map_obj = create_folium_map(gdf_merged, key_column='join_name', tooltip_name_col='display_name', tooltip_prov_col='prov')
                    if map_obj:
                        st.session_state['map_object'] = map_obj
                    else:
                        st.warning("Gagal membuat objek peta interaktif dari GeoDataFrame hasil merge.")

                except FileNotFoundError as fe:
                    st.error(str(fe))
                    st.session_state['map_object'] = None
                except Exception as e:
                    st.error(f"Gagal memuat GeoJSON atau membuat peta: {e}")
                    st.session_state['map_object'] = None

            else:
                st.warning("Labels clustering tidak tersedia atau panjang tidak sesuai dengan data.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")
        st.exception(e)
        st.session_state['hasil_data'] = None
        st.session_state['gdf_hasil'] = None
        st.session_state['scores'] = None
        st.session_state['data_for_clustering'] = None
        st.session_state['map_object'] = None
        st.session_state['var'] = None
        st.session_state['params'] = None
        
def _detect_name_column(gdf: gpd.GeoDataFrame, candidates: Optional[list] = None) -> Optional[str]:
    """
    Try to detect the column in gdf that contains the district/kabupaten/kota name.
    Returns the column name or None if not found.
    """
    if candidates is None:
        candidates = [
            "WADMKK", "WADMKKK", "WADMKK_", "NAME_2", "NAME_1", "NAMA_KAB", "NAMA", "NM_KAB",
            "nm_kab", "KABUPATEN", "KAB", "KAB_KOTA", "KABKOTA", "KOTA", "KABKOT", "kab_kota",
            "KAB_CODE", "KAB_KODE", "NM_KEC", "KAB_NAMA", "district", "district_name"
        ]
    for c in candidates:
        if c in gdf.columns:
            return c
    for col in gdf.columns:
        col_l = col.lower()
        if ("kab" in col_l) or ("kota" in col_l) or ("name" in col_l) or ("nama" in col_l) or ("district" in col_l):
            return col
    return None


def _detect_prov_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Detect column containing province name if exists.
    """
    for col in gdf.columns:
        col_l = col.lower()
        if ("prov" in col_l) or ("provinsi" in col_l) or ("province" in col_l) or ("prov_name" in col_l) or ("wadmpr" in col_l):
            return col
    return None


def _normalize_name_series(s: pd.Series) -> pd.Series:
    """
    Normalize names:
    - uppercase, strip
    - replace 'KABUPATEN ' -> 'KAB. ' and 'KEPULAUAN ' -> 'KEP. '
    - replace adm / adm. / ADM / ADM. -> ADMINISTRASI
    - remove duplicate spaces
    Returns normalized series (still containing spaces).
    """
    s = s.astype(str).fillna("").str.upper().str.strip()
    s = s.str.replace("KABUPATEN ", "KAB. ", regex=False)
    s = s.str.replace("KEPULAUAN ", "KEP. ", regex=False)
    s = s.str.replace(r"\bADM\.?\b", "ADMINISTRASI", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _make_join_key(s: pd.Series) -> pd.Series:
    """
    Create a join key by removing spaces and non-word characters so both sides can be merged.
    This "joined" key is used for matching; we keep the original display name for tooltips.
    """
    key = s.astype(str).fillna("").str.upper()
    key = key.str.replace(r"[^\w]", "", regex=True)
    key = key.str.strip()
    return key