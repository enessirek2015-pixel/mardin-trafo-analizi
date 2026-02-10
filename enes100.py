import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from scipy.spatial import KDTree
import numpy as np

# Sayfa Ayarları
st.set_page_config(page_title="Trafo Network Analizi", layout="wide")

st.title("⚡ Trafo Yakınlık ve Network Analizi")
st.markdown("""
Bu uygulama ile trafo verilerinizi yükleyebilir ve belirlediğiniz mesafe kriterine göre şebeke yakınlık analizi yapabilirsiniz.
""")

# --- YAN PANEL: DOSYA YÜKLEME ---
st.sidebar.header("📂 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader("Excel dosyasını seçiniz (.xlsx)", type=["xlsx"])


@st.cache_data
def process_data(file):
    try:
        df = pd.read_excel(file)
        # Sütun isimlerindeki boşlukları temizle
        df.columns = df.columns.str.strip()

        # Gerekli sütun kontrolü
        required = ['TRAFO_KODU', 'İLÇE', 'CBS_X', 'CBS_Y', 'ABONE_GRUP_ADI']
        missing = [col for col in required if col not in df.columns]

        if missing:
            return None, f"Hata: Eksik sütunlar var: {', '.join(missing)}"

        # Temizlik
        df = df.dropna(subset=['CBS_X', 'CBS_Y', 'TRAFO_KODU'])
        df_unique = df.drop_duplicates(subset=['TRAFO_KODU']).copy()
        return df_unique, None
    except Exception as e:
        return None, f"Dosya okunurken bir hata oluştu: {e}"


if uploaded_file is not None:
    data, error = process_data(uploaded_file)

    if error:
        st.error(error)
    else:
        # --- FİLTRELER ---
        st.sidebar.header("📍 Analiz Parametreleri")

        ilceler = sorted(data['İLÇE'].unique())
        secilen_ilce = st.sidebar.selectbox("İlçe Seçin", ilceler)

        mesafe_siniri = st.sidebar.slider("Yakınlık Mesafesi (Metre)", 50, 3000, 500)

        # Veriyi Filtrele
        filtered_df = data[data['İLÇE'] == secilen_ilce].reset_index(drop=True)
        st.sidebar.success(f"Analiz Edilen Trafo Sayısı: {len(filtered_df)}")

        if len(filtered_df) > 0:
            # --- HESAPLAMA ---
            lat_avg = filtered_df['CBS_Y'].mean()
            lon_avg = filtered_df['CBS_X'].mean()

            coords = filtered_df[['CBS_Y', 'CBS_X']].values
            coords_m = coords.copy()
            coords_m[:, 0] = coords[:, 0] * 111320
            coords_m[:, 1] = coords[:, 1] * 111320 * np.cos(np.radians(lat_avg))

            tree = KDTree(coords_m)
            yakin_noktalar = tree.query_pairs(mesafe_siniri)

            # --- HARİTA ---
            m = folium.Map(location=[lat_avg, lon_avg], zoom_start=13, tiles='OpenStreetMap')

            # Bağlantılar
            for i, j in yakin_noktalar:
                p1 = filtered_df.iloc[i]
                p2 = filtered_df.iloc[j]
                folium.PolyLine(
                    locations=[[p1['CBS_Y'], p1['CBS_X']], [p2['CBS_Y'], p2['CBS_X']]],
                    color="red", weight=2, opacity=0.7
                ).add_to(m)

            # Nodlar
            for idx, row in filtered_df.iterrows():
                folium.CircleMarker(
                    location=[row['CBS_Y'], row['CBS_X']],
                    radius=5, color='blue', fill=True,
                    popup=f"Trafo: {row['TRAFO_KODU']}<br>Grup: {row['ABONE_GRUP_ADI']}"
                ).add_to(m)

            st_folium(m, width=1200, height=700, returned_objects=[])
        else:
            st.warning("Seçilen ilçede veri bulunamadı.")
else:
    st.info("Lütfen sol panelden bir Excel dosyası yükleyerek analizi başlatın.")
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)  # Görsel bir dokunuş