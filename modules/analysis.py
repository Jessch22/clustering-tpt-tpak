import streamlit as st
import pandas as pd
import geopandas as gpd
import time
from modules.data_processing import muat_data, preprocessing_data, kmeans_clustering, dbscan_clustering
from modules.plot import create_folium_map

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

def run_analysis(var, tahun_pilihan, metode_terpilih, params, path, sheet):
    """Fungsi untuk Menjalankan analisis clustering dari logic/computation.py"""
    
    # Menyimpan: hasil data yang telah diproses, skor metrik, gdf hasil, data untuk clustering (sudah dinormalisasi)
    st.session_state['hasil_data'] = None
    st.session_state['scores'] = None
    st.session_state['gdf_hasil'] = None
    st.session_state['data_for_clustering'] = None
    st.session_state['map_object'] = None
    st.session_state['cluster_color_map'] = None
    
    try:
        # Mulai pengukuran waktu
        # Inisialisasi session state
        # Menyimpan: hasil data yang telah diproses, skor metrik, gdf hasil, data untuk clustering (sudah dinormalisasi)
        st.session_state['hasil_data'] = None
        st.session_state['scores'] = None
        st.session_state['gdf_hasil'] = None
        st.session_state['data_for_clustering'] = None
        st.session_state['map_object'] = None
        st.session_state['cluster_color_map'] = None
        
        start_proc = time.perf_counter()

        # ============ 1. Muat dan Preprocessing Data ============
        df_raw = muat_data(path, sheet)
        
        # Berhenti jika data kosong atau error
        if df_raw.empty: 
            return 
            return
        
        # Simpan kolom identitas: ID, prov, kab_kota
        identity_cols = df_raw[['ID', 'prov', 'kab_kota']].copy()
        
        # Preprocessing data
        datacol_num, data_replace, data_clean, data_norm, data_splits = preprocessing_data(df_raw)
        
        # ========================================================

        # ============ 2. Pilih data untuk clustering ============
        start_year = tahun_pilihan[0]
        end_year = tahun_pilihan[1]
        # Buat nama data sesuai var dan tahun
        nama_data = f"{var}_{start_year}" if start_year == end_year else f"{var}_{start_year}_{end_year}"
        
        # Cek nama_data ada di data_splits atau ga
        if nama_data not in data_splits:
            st.error(f"Data split '{nama_data}' tidak ditemukan.")
            return
        
        # Simpan data untuk clustering
        data_for_clustering = data_splits[nama_data]
        # Simpan di session state
        st.session_state['data_for_clustering'] = data_for_clustering

        # ========================================================

        # ================ 3. Lakukan Clustering =================
        # Inisialisasi variabel
        labels = None       #Label cluster
        point_type = None   #Tipe titik (untuk DBSCAN)
        sil_score = None    #Silhouette Score
        dbi_score = None    #Davies-Bouldin Index
        
        # K-MEANS
        with st.spinner("Menjalankan clustering..."):
            if metode_terpilih == "K-Means":
                k = params['k']
                # Menjalankan K-Means
                hasil_cluster = kmeans_clustering(data_for_clustering, k)
                
                # Ambil hasil: labels, sil_score, dbi_score
                labels = hasil_cluster['labels']
                sil_score = hasil_cluster.get('silhouette', None)
                dbi_score = hasil_cluster.get('dbi', None)
                
                # Debug K-Means selesai
                st.success(f"K-Means selesai. Jumlah label: {len(labels) if labels is not None else 'None'}")
                

            # DBSCAN
            elif metode_terpilih == "DBSCAN":
                eps = params['eps']
                minpts = params['minpts']
                
                # Menjalankan DBSCAN
                hasil_cluster = dbscan_clustering(data_for_clustering, eps, minpts)
                
                # Ambil hasil: labels, point_type, sil_score, dbi_score
                labels = hasil_cluster['labels']
                point_type = hasil_cluster.get('point_type', None)
                sil_score = hasil_cluster.get('silhouette', None)
                dbi_score = hasil_cluster.get('dbi', None)
                
                # Debug DBSCAN selesai
                st.success(f"DBSCAN selesai. Jumlah label: {len(labels) if labels is not None else 'None'}. Jumlah point_type: {len(point_type) if point_type is not None else 'None'}") # DEBUG
            
        # ========================================================
        
        # ============== 4. Buat Tabel Hasil Akhir ===============
        # Jika labels ada, buat tabel hasil akhir
        with st.spinner("Membuat visualisasi..."):
            if labels is not None:
                if data_clean.shape[0] == len(labels):
                    # Gabung kolom identitas dengan data_clean ke df_output_temp
                    df_output_temp = pd.concat([identity_cols.reset_index(drop=True), data_clean.reset_index(drop=True)], axis=1)
                    
                    # Tambah kolom Cluster ke df_output_temp
                    df_output_temp['Cluster'] = labels
                    
                    # Tambah kolom Point Type ke df_output_temp jika DBSCAN
                    if metode_terpilih == "DBSCAN":
                        if point_type is not None and len(point_type) == data_clean.shape[0]:
                            df_output_temp['Point Type'] = point_type
                        else:
                            # Debug gagal tambah Point Type
                            st.warning(f"Gagal menambahkan 'Point Type'. point_type: {'Ada' if point_type is not None else 'None'}, len(point_type): {len(point_type) if point_type is not None else 'N/A'}, data_clean.shape[0]: {data_clean.shape[0]}")
                            df_output_temp['Point Type'] = None
                            
                    # Simpan ke session state
                    st.session_state['hasil_data'] = df_output_temp
                    
                    # Simpan skor metrik dan waktu proses
                    # Hitung waktu proses & simpan skor metrik: Silhouette Score, Davies-Bouldin Index
                    end_proc = time.perf_counter()
                    elapsed_sec = end_proc - start_proc
                    st.session_state['scores'] = {'silhouette': sil_score, 'dbi': dbi_score, 'time_sec': elapsed_sec}
                    
                    # Peta Hasil Clustering
                    # Simpan peta kosong di session state
                    st.session_state['map_object'] = None
                    
                    try:
                        file_id = "1PmftUhdE8eXgexmAUd01BxZj8Mm6S0Jf"
                        # Muat shapefile batas kabupaten/kota
                        shapefile_path = f'https://drive.google.com/uc?export=download&id={file_id}'
                        gdf = gpd.read_file(shapefile_path)
                        # Penyederhanaan geometri
                        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)
                        
                        NAMA_KAB_KOTA = 'WADMKK'
                        if NAMA_KAB_KOTA not in gdf.columns:
                            st.error(f"Kolom '{NAMA_KAB_KOTA}' tidak ditemukan di Shapefile.")
                        else:
                            # Normalisasi nama kab_kota di gdf dan df_hasil_final_map
                            # Ubah: jadi uppercase, hilangkan spasi di awal/akhir, ganti "KABUPATEN " jadi "KAB. ", ganti "KEPULAUAN " jadi "KEP. "
                            gdf[NAMA_KAB_KOTA] = gdf[NAMA_KAB_KOTA].astype(str).str.upper().str.strip().str.replace("KABUPATEN ", "KAB. ", regex=False).str.replace("KEPULAUAN ", "KEP. ", regex=False)
                            df_hasil_final_map = st.session_state['hasil_data'].copy()
                            df_hasil_final_map['kab_kota'] = df_hasil_final_map['kab_kota'].astype(str).str.upper().str.strip().str.replace("KABUPATEN ", "KAB. ", regex=False).str.replace("KEPULAUAN ", "KEP. ", regex=False)
                            
                            # Pastikan kolom 'prov' bertipe str jika ada
                            if 'prov' in df_hasil_final_map.columns:
                                df_hasil_final_map['prov'] = df_hasil_final_map['prov'].astype(str)
                            
                            # Kolom untuk di-merge: kab_kota dan Cluster
                            cols_to_merge = ['kab_kota', 'Cluster']
                            # Tambah kolom 'prov' jika ada
                            if 'prov' in df_hasil_final_map.columns:
                                cols_to_merge.append('prov')

                            # Lakukan merge: gdf dengan df_hasil_final_map berdasarkan kolom 'kab_kota' dan 'Cluster'
                            gdf_merged = gdf.merge(
                                df_hasil_final_map[cols_to_merge],
                                left_on=NAMA_KAB_KOTA,
                                right_on='kab_kota',
                                how='left'
                            )
                            
                            # Simpan gdf hasil di session state
                            st.session_state['gdf_hasil'] = gdf_merged
                            
                            # Cek berapa wilayah yang berhasil di-merge
                            n_matched = gdf_merged['Cluster'].notna().sum()
                            n_total = len(gdf_merged)
                            
                            # Buat Peta Folium dari hasil merge
                            map_obj = create_folium_map(gdf_merged, key_column=NAMA_KAB_KOTA)
                            if map_obj:
                                st.session_state['map_object'] = map_obj
                            else:
                                st.write("Gagal membuat objek peta interaktif.")
                    
                    # Jika terjadi kesalahan saat memuat shapefile atau pemrosesan peta
                    except Exception as e:
                        st.error(f"Gagal memuat shapefile atau membuat peta: {e}")
                        st.session_state['map_object'] = None


    except Exception as e:
        # Jika terjadi kesalahan saat memproses
        # Simpan hasil data, peta, skor metrik, dan data untuk clustering di session state sebagai None
        st.error(f"Terjadi kesalahan saat memproses: {e}")
        st.exception(e)
        st.session_state['hasil_data'] = None
        st.session_state['gdf_hasil'] = None
        st.session_state['scores'] = None
        st.session_state['data_for_clustering'] = None
        st.session_state['map_object'] = None
        st.session_state['var'] = None
        st.session_state['params'] = None