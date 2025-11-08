import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# from utils.saveaspdf import generate_pdf_report

#* 1. Membaca data
def muat_data(path, sheet):
  try:
    # Membaca data
    df = pd.read_excel(path, sheet_name=sheet, dtype={'kab_kota': str})
    # Ganti nilai NaN/None pada kolom 'kab_kota' (jika ada)
    df['kab_kota'] = df['kab_kota'].fillna('').astype(str)
  except ValueError as e:
      # Jika kolom 'kab_kota' tidak ada, baca seperti biasa
      if "Unknown column name 'kab_kota'" in str(e):
          st.warning("Kolom 'kab_kota' tidak ditemukan saat memuat data awal. Pastikan nama kolom benar di Excel.")
          df = pd.read_excel(path, sheet_name=sheet)
      else:
          raise e
  # Jika gagal membaca file
  except Exception as e:
      st.error(f"Gagal memuat data dari {path} sheet {sheet}: {e}")
      return pd.DataFrame()

  return df

#* 2. Preprocessing
# Menghapus non-numerik
def del_col_non_numeric(data):
  col_non_numeric = ['prov', 'kab_kota']
  cek_column = [c for c in col_non_numeric if c in data.columns]
  data_num = data.drop(columns=cek_column)
  return data_num

# Mengganti non-numeric dengan NaN
def replace_non_numeric(data):
  data_replaced = data.replace({None: np.nan, '-': np.nan})
  data_num = data.replace({None: np.nan, '-': np.nan})
  return data_num

# Mengisi missing value
def missing_value(data):
  data_clean = data.copy()
  # Cek missing value
  if data_clean.isnull().sum().sum() > 0:
    list_indikator = list(set(kolom.split('_')[0] for kolom in data_clean.columns))
    
    for indikator in list_indikator:
      kolom_tiap_tahun = [k for k in data_clean.columns if k.startswith(indikator + '_')]
      kolom_tiap_tahun = sorted(kolom_tiap_tahun, key=lambda x: int(x.split('_')[1]))
      data_tiap_tahun = data_clean[kolom_tiap_tahun]
      # Interpolasi linear untuk missing di tengah
      data_tiap_tahun = data_tiap_tahun.interpolate(axis=1, method='linear')
      # Backfill untuk missing di depan
      data_tiap_tahun = data_tiap_tahun.bfill(axis=1)
      # Forward fill untuk missing di belakang
      data_tiap_tahun = data_tiap_tahun.ffill(axis=1)

      data_clean[kolom_tiap_tahun] = data_tiap_tahun
  return data_clean

# Normalisasi
def data_normalization(data):
  data_norm = data.copy()

  for col in data_norm.columns:
    mean = data_norm[col].mean()
    std = data_norm[col].std()
    data_norm[col] = (data_norm[col] - mean) / std
  
  return data_norm

# Bagi data
def bagi_data(data):

  data_splits = {}
  tahun_list = list(range(2018, 2025))

  for i in range(len(tahun_list)):
    for j in range(i, len(tahun_list)):
      start = tahun_list[i]
      end = tahun_list[j]
      rentang = "|".join(str(t) for t in range(start, end + 1))
      suffix = f"{start}_{end}" if start != end else str(start)
      
      # Split kolom berdasarkan rentang tahun
      data_splits[f"tpt_tpak_{suffix}"] = data.filter(regex=f"^(TPT|TPAK)_({rentang})")
      # Split untuk masing-masing indikator
      data_splits[f"tpt_{suffix}"] = data.filter(regex=f"^TPT_({rentang})")
      data_splits[f"tpak_{suffix}"] = data.filter(regex=f"^TPAK_({rentang})")
  return data_splits

# Preprocessing Keseluruhan
def preprocessing_data(data):
  datacol_num = del_col_non_numeric(data)
  data_replace = replace_non_numeric(datacol_num)
  data_clean = missing_value(data_replace)
  data_norm = data_normalization(data_clean)
  data_splits = bagi_data(data_norm)
  return datacol_num, data_replace, data_clean, data_norm, data_splits

#* 3. K-Means
def kmeans_clustering(data, nilai_k):
  # Cek apakah data dan nilai k telah dipilih
  if data is None or nilai_k is None:
    st.info("Variabel/Tahun belum dipilih")
    return
  
  # Menjalankan K-Means
  kmeans = KMeans(n_clusters=nilai_k, random_state=42)
  labels = kmeans.fit_predict(data)
  
  return {"labels": labels, "centroids": kmeans.cluster_centers_}
  
#* 4. DBSCAN
def dbscan_clustering(data, eps, min_samples):
    # Menjalankan DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
    
    # Mendapatkan label dan point type
    labels = dbscan.fit_predict(data)
    core_indices = dbscan.core_sample_indices_
    
    point_type = np.full(len(labels), "Border", dtype=object)
    point_type[core_indices] = "Core"
    point_type[labels == -1] = "Noise"
    
    
    return {
        "labels": labels,
        "point_type": point_type.tolist(),
    }