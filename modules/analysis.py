import streamlit as st
import pandas as pd
import geopandas as gpd
import time
import os
from modules.data_processing import muat_data, preprocessing_data, kmeans_clustering, dbscan_clustering
from modules.plot import create_folium_map
from typing import Optional
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_analysis(var, tahun_pilihan, metode_terpilih, params, path, sheet, logger):
    """Fungsi untuk Menjalankan analisis clustering dari logic/computation.py"""

    # Initialize/clear relevant session state keys
    st.session_state['hasil_data'] = None
    st.session_state['scores'] = None
    st.session_state['gdf_hasil'] = None
    st.session_state['data_for_clustering'] = None
    st.session_state['map_object'] = None
    st.session_state['cluster_color_map'] = None
    
    st.session_state['var'] = var

    start_proc = time.perf_counter()

    try:
        # ============ 1. Muat dan Preprocessing Data ============
        logger.info("Memuat dan Preprocessing Data...")
        df_raw = muat_data(path, sheet)
        if df_raw.empty:
            logger.error("Dataset kosong. Pastikan file DATASET.xlsx benar dan memiliki sheet yang dipilih.")
            st.error("Dataset kosong. Pastikan file DATASET.xlsx benar dan memiliki sheet yang dipilih.")
            return
        else:
            logger.info("Dataset berhasil dimuat.")

        # Pastikan ada kolom prov, kab_kota
        required_cols = ['prov', 'kab_kota']
        for c in required_cols:
            if c not in df_raw.columns:
                logger.warning(f"Kolom '{c}' tidak ditemukan di dataset. Pastikan dataset memiliki kolom ID, prov, kab_kota.")
                return
            else:
                logger.info(f"Kolom '{c}' ditemukan.")

        identity_cols = df_raw[['prov', 'kab_kota']].copy()

        # Preprocessing: menghasilkan data_clean, data_norm, data_splits, dsb.
        datacol_num, data_replace, data_clean, data_norm, data_splits = preprocessing_data(df_raw)

        # ============ 2. Pilih data untuk clustering ============
        start_year = tahun_pilihan[0]
        end_year = tahun_pilihan[1]
        nama_data = f"{var}_{start_year}" if start_year == end_year else f"{var}_{start_year}_{end_year}"
        logger.info(f"Data dipilih: {nama_data} (Dimensi: {data_splits.get(nama_data, pd.DataFrame()).shape})")

        if nama_data not in data_splits:
            st.error(f"Data split '{nama_data}' tidak ditemukan.")
            logger.error(f"Data split '{nama_data}' tidak ditemukan.")
            return
        else:
            logger.info(f"Data split '{nama_data}' ditemukan.")

        data_for_clustering = data_splits[nama_data]
        st.session_state['data_for_clustering'] = data_for_clustering
        
        D = data_for_clustering.shape[1]
        data_to_cluster = data_for_clustering.copy()

        # ================ 3. Lakukan Clustering =================
        labels = None
        point_type = None
        sil_score = None
        dbi_score = None

    
        if metode_terpilih == "K-Means":
            logger.info("Menjalankan K-Means...")
            optimal_k = params.get('optimal_k', False)
            
            if optimal_k:
                logger.info(f"Mencari K optimal menggunakan Silhouette Score...")
                best_k = 2
                best_score = -1
                k_range = range(2, 7)
                
                for k_test in k_range:
                    temp_result = kmeans_clustering(data_to_cluster, k_test)
                    temp_labels = temp_result.get('labels')
                    if temp_labels is not None and len(np.unique(temp_labels)) > 1:
                        temp_sil = silhouette_score(data_to_cluster, temp_labels) 
                        if temp_sil > best_score:
                            best_score = temp_sil
                            best_k = k_test
                            
                k = best_k
                params['k'] = k
                st.info(f"K optimal ditemukan: {k} (Silhouette: {best_score:.4f})")
                logger.info(f"K optimal ditemukan: {k} (Silhouette: {best_score:.4f})")
            else:
                k = params.get('k', 2)
            
            hasil_cluster = kmeans_clustering(data_to_cluster, k)
            labels = hasil_cluster.get('labels')
                    
        elif metode_terpilih == "DBSCAN":
            logger.info("Menjalankan DBSCAN...")
            optimal_dbscan = params.get('optimal_dbscan', False)
            use_pca_manual = params.get('use_pca_manual', False)
            
            D_final = D
            data_to_cluster = data_for_clustering.copy() # Mulai dengan data asli
            pca_applied = False
            
            run_pca = False
            if optimal_dbscan and D >= 3:
                run_pca = True
                logger.info("Mode optimal: PCA akan diterapkan (D>=3).")
            elif not optimal_dbscan and use_pca_manual and D >= 3:
                run_pca = True
                logger.info("Mode manual: PCA akan diterapkan (sesuai pilihan user).")
            elif not optimal_dbscan and not use_pca_manual:
                logger.info("Mode manual: PCA TIDAK diterapkan. Clustering pada data dimensi penuh.")
            else: 
                logger.info(f"Dimensi asli ({D}) < 3. PCA tidak diterapkan.")
            
            if run_pca:
                n_components_pca = 3 if D > 3 else 2 
                pca = PCA(n_components=n_components_pca)
                data_pca = pca.fit_transform(data_to_cluster)
                logger.info(f"Data ditransformasi ke {n_components_pca} komponen utama.")
                data_to_cluster = pd.DataFrame(data_pca, index=data_for_clustering.index, columns=[f"PC{i+1}" for i in range(n_components_pca)])
                D_final = n_components_pca
                pca_applied = True
                
            if optimal_dbscan:
                logger.info("Mencari parameter optimal...")                
                minpts = max(2, D_final + 1)
                logger.info(f"Dimensi final: {D_final}. MinPts diatur ke max(3, D_final+1) = {minpts}")
                
                nn = NearestNeighbors(n_neighbors=minpts, metric='manhattan') 
                nn.fit(data_to_cluster)
                distances, indices = nn.kneighbors(data_to_cluster)
                k_distances = np.sort(distances[:, -1], axis=0)
                
                x = np.arange(len(k_distances))
                y = k_distances
                
                eps = 0.5
                try:
                    kneedle = KneeLocator(x, y, curve='convex', direction='increasing', S=1.0)
                    eps = kneedle.elbow_y
                    if eps is None or eps <= 0:
                        logger.warning(f"Kneed gagal (eps={eps}). Fallback ke 0.5")
                        eps = 0.5
                except Exception as e_kneed:
                    logger.warning(f"Error Kneed: {e_kneed}. Fallback ke 0.5")
                    eps = 0.5
                    
                SLIDER_MIN = 0.1
                SLIDER_MAX = 25.0
                SLIDER_STEP = 0.1
                
                eps_rounded = np.round(eps / SLIDER_STEP) * SLIDER_STEP
                eps_final = np.clip(eps_rounded, SLIDER_MIN, SLIDER_MAX)
                
                eps = eps_final
                
                st.success(f"Epsilon optimal: {eps:.2f}, MinPts: {minpts}")
                logger.success(f"Epsilon optimal: {eps:.2f}, MinPts: {minpts}")

                # Simpan params optimal
                params['eps'] = eps
                params['minpts'] = minpts
                
            else:
                # Ambil dari slider
                eps = params.get('eps', 0.5)
                minpts = params.get('minpts', 5)
                logger.info(f"Parameter: MinPts = {minpts}, Epsilon = {eps}")
                
            # Panggil clustering dengan params final
            # data_to_cluster bisa jadi data asli atau data PCA
            hasil_cluster = dbscan_clustering(data_to_cluster, eps, minpts) 
            labels = hasil_cluster.get('labels')
            point_type = hasil_cluster.get('point_type', None)

        else:
            st.error("Metode clustering tidak dikenali.")
            logger.error("Metode clustering tidak dikenali")
            return
            
        logger.info("Menghitung skor evaluasi...")
        if labels is not None and len(np.unique(labels)) > 1:
            sil_score = silhouette_score(data_for_clustering, labels)
            dbi_score = davies_bouldin_score(data_for_clustering, labels)
            logger.info(f"Skor: Sil = {sil_score:.4f}, DBI = {dbi_score:.4f}")
        else:
            sil_score = None
            dbi_score = None
            logger.warning("Skor tidak dihitung (kurang dari 2 cluster).")
            
        # ============== 4. Buat Tabel Hasil Akhir ===============
        logger.info("Membuat tabel hasil dan visualisasi...")
        if labels is not None and data_clean.shape[0] == len(labels):
            df_output_temp = pd.concat([identity_cols.reset_index(drop=True), data_clean.reset_index(drop=True)], axis=1)
            df_output_temp['Cluster'] = labels

            if metode_terpilih == "DBSCAN":
                if point_type is not None and len(point_type) == data_clean.shape[0]:
                    df_output_temp['Point Type'] = point_type
                else:
                    st.warning("Gagal menambahkan kolom 'Point Type' (DBSCAN). Akan diisi None.")
                    logger.warning("Gagal menambahkan kolom 'Point Type' (DBSCAN). Akan diisi None.")
                    df_output_temp['Point Type'] = None

            st.session_state['hasil_data'] = df_output_temp

            end_proc = time.perf_counter()
            elapsed_sec = end_proc - start_proc
            
            st.session_state['scores'] = {'silhouette': sil_score, 'dbi': dbi_score, 'time_sec': elapsed_sec}
            st.session_state['params'] = params

            try:
                logger.info("Memuat GeoJSON...")
                geojson_path = r'geojson/38 Provinsi Indonesia - Kabupaten.json'
                if not os.path.exists(geojson_path):
                    raise FileNotFoundError(f"GeoJSON tidak ditemukan. Letakkan file GeoJSON di: {geojson_path}")

                gdf = gpd.read_file(geojson_path)

                NAMA_KAB_KOTA = _detect_name_column(gdf)
                if NAMA_KAB_KOTA is None:
                    logger.error("Tidak dapat mendeteksi kolom nama wilayah di GeoJSON. Kolom yang biasa: WADMKK, NAME_2, NAMA, dsb.")
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
                    logger.info("Melakukan merge data Peta...")
                    gdf_merged = gdf.merge(
                        df_hasil_final_map[['join_name', 'Cluster', 'prov_d', 'orig_name_df']],
                        left_on='join_name',
                        right_on='join_name',
                        how='left',
                        validate="m:1"
                    )

                except Exception as e_merge:
                    logger.warning(f"Merge Peta (m:1) gagal: {e_merge}. Fallback ke merge 'left'.")
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
                    unmatched = gdf_merged[gdf_merged['Cluster'].isna()][['orig_name', 'prov']]
                    if not unmatched.empty:
                        logger.warning(f"{len(unmatched)} wilayah di peta tidak cocok dengan data.")
                except Exception as e_unmatched:
                    logger.warning(f"Gagal mengecek wilayah unmatched: {e_unmatched}")
                try:
                    n_matched = gdf_merged['Cluster'].notna().sum()
                    n_total = len(gdf_merged)
                    logger.success(f"Wilayah yang berhasil dicocokkan: {n_matched-1}/{n_total-1}")
                except Exception:
                    pass
                
                logger.info("Membuat objek Peta Folium...")
                map_obj = create_folium_map(gdf_merged, key_column='join_name', tooltip_name_col='display_name', tooltip_prov_col='prov')
                if map_obj:
                    st.session_state['map_object'] = map_obj
                    logger.success("Peta Folium berhasil dibuat.")
                else:
                    logger.warning("Gagal membuat objek peta interaktif dari GeoDataFrame hasil merge.")

            except FileNotFoundError as fe:
                logger.error(str(fe))
                st.session_state['map_object'] = None
            except Exception as e:
                logger.error(f"Gagal memuat GeoJSON atau membuat peta: {e}")
                st.session_state['map_object'] = None

        else:
            logger.warning("Labels clustering tidak tersedia atau panjang tidak sesuai dengan data.")

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memproses: {e}")
        logger.exception(e)
        st.session_state['hasil_data'] = None
        st.session_state['gdf_hasil'] = None
        st.session_state['scores'] = None
        st.session_state['data_for_clustering'] = None
        st.session_state['map_object'] = None
        st.session_state['var'] = None
        st.session_state['params'] = None
        
    logger.success("Analisis selesai.")
        
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