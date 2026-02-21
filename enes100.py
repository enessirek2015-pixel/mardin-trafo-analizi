
"""
RF Drive-By: Saha İş Emri ve Akıllı Durak Planlama
Gelişmiş Versiyon — Daha iyi UX, zengin analiz, temiz mimari
"""

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
from streamlit_folium import st_folium
from scipy.spatial import KDTree
import numpy as np
import networkx as nx
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📡 RF Saha Analiz Merkezi",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STİL (CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #fff;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "📡";
    font-size: 8rem;
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0.08;
  }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .hero p { color: #b0b8d0; margin: 0.4rem 0 0; font-size: 0.95rem; }

  .metric-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #fff;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
  }
  .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }
  .metric-card .value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
  .metric-card .label { font-size: 0.8rem; color: #7c8399; margin-top: 0.2rem; }

  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e0e4f5;
    padding: 0.5rem 0;
    border-bottom: 2px solid #302b63;
    margin: 1rem 0 0.8rem;
  }

  .info-box {
    background: linear-gradient(90deg, #1a1f3a, #1e2130);
    border-left: 4px solid #636bff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    color: #c0c8e0;
    font-size: 0.88rem;
    margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SABİT DEĞERLER
# ─────────────────────────────────────────────────────────────────────────────
AYLAR = ['OCAK','ŞUBAT','MART','NİSAN','MAYIS','HAZİRAN',
         'TEMMUZ','AĞUSTOS','EYLÜL','EKİM','KASIM','ARALIK']

ABONE_ONCELIK = {
    'TİCARETHANE': 10, 'RESMİ DAİRELER': 10, 'BELEDİYELER': 10,
    'OSB/JEOTERMAL': 10, 'MESKEN': 5, 'CAMİ - İBADETHANE': 5,
    'KİTLER': 5, 'TARIMSAL SULAMA': 1, 'ŞANTİYE': 1, 'ICME SUYU': 1
}

HARITA_TIPLERI = {
    "🛰️ Uydu": ('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 'Google Satellite'),
    "🗺️ Topoğrafik": ('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', 'OpenTopoMap'),
    "🧾 Sade": ('cartodbpositron', 'CartoDB'),
    "🌙 Koyu": ('cartodbdark_matter', 'CartoDB Dark'),
}

# ─────────────────────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📊 Veri işleniyor...")
def load_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = df.columns.str.strip()

        for c in ['TRAFO_KODU', 'İLÇE', 'MAHALLE', 'ABONE_GRUP_ADI']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        df['CBS_X'] = pd.to_numeric(df['CBS_X'], errors='coerce').astype(float)
        df['CBS_Y'] = pd.to_numeric(df['CBS_Y'], errors='coerce').astype(float)

        for c in ['GİDİLME SAYISI'] + [m for m in AYLAR if m in df.columns]:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

        return (df.dropna(subset=['CBS_X', 'CBS_Y', 'TRAFO_KODU'])
                  .drop_duplicates(subset=['TRAFO_KODU'])
                  .reset_index(drop=True))
    except Exception as e:
        st.error(f"❌ Veri yükleme hatası: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER HESAPLA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔗 RF ağı hesaplanıyor...")
def compute_clusters(df_json: str, rf_menzil: int):
    df = pd.read_json(df_json, orient='split')
    lat_avg = float(df['CBS_Y'].mean())
    coords = df[['CBS_Y', 'CBS_X']].values.astype(float)
    coords_m = coords.copy()
    coords_m[:, 0] *= 111320
    coords_m[:, 1] *= 111320 * np.cos(np.radians(lat_avg))

    tree = KDTree(coords_m)
    pairs = list(tree.query_pairs(rf_menzil))
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    G.add_edges_from(pairs)
    return pairs, list(nx.connected_components(G))


# ─────────────────────────────────────────────────────────────────────────────
# HARİTA OLUŞTUR
# ─────────────────────────────────────────────────────────────────────────────
def create_map(df, pairs, clusters, katman, secilen_ay, show_heatmap, show_mesh):
    lat_avg = float(df['CBS_Y'].mean())
    lon_avg = float(df['CBS_X'].mean())
    tiles, attr = HARITA_TIPLERI[katman]
    dark = "Koyu" in katman

    m = folium.Map(location=[lat_avg, lon_avg], zoom_start=14,
                   tiles=tiles, attr=attr, prefer_canvas=True)
    MiniMap(toggle_display=True).add_to(m)

    random.seed(42)
    palette = [f"#{random.randint(0x303060, 0xCCDDFF):06x}" for _ in range(len(clusters))]
    cizgi = "rgba(255,255,255,0.4)" if dark else "rgba(50,50,180,0.35)"

    # Heatmap
    if show_heatmap and 'GİDİLME SAYISI' in df.columns:
        heat_data = [
            [float(r['CBS_Y']), float(r['CBS_X']), float(r['GİDİLME SAYISI']) + 1]
            for _, r in df.iterrows()
        ]
        HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3,
                gradient={'0.4': 'blue', '0.65': 'lime', '1': 'red'}).add_to(m)

    # Mesh çizgileri
    if show_mesh:
        for i, j in pairs:
            p1, p2 = df.iloc[i], df.iloc[j]
            folium.PolyLine(
                [[float(p1['CBS_Y']), float(p1['CBS_X'])],
                 [float(p2['CBS_Y']), float(p2['CBS_X'])]],
                color=cizgi, weight=1.5, opacity=0.7
            ).add_to(m)

    # Durak + Trafo işaretçileri
    for cid, nodes in enumerate(clusters):
        color = palette[cid] if len(nodes) > 1 else "#888899"
        cluster_df = df.iloc[list(nodes)].copy()

        if len(nodes) > 1:
            cluster_df['puan'] = cluster_df['ABONE_GRUP_ADI'].map(ABONE_ONCELIK).fillna(1)
            best = cluster_df[cluster_df['puan'] == cluster_df['puan'].max()]
            cx = float(cluster_df['CBS_X'].mean())
            cy = float(cluster_df['CBS_Y'].mean())
            dists = np.hypot(best['CBS_Y'].astype(float) - cy,
                             best['CBS_X'].astype(float) - cx)
            stop = best.loc[dists.idxmin()]

            toplam_gidilme = int(cluster_df['GİDİLME SAYISI'].sum()) if 'GİDİLME SAYISI' in cluster_df.columns else 0
            stop_lat = float(stop['CBS_Y'])
            stop_lon = float(stop['CBS_X'])
            stop_tur = str(stop['ABONE_GRUP_ADI'])

            aylik_html = ""
            if secilen_ay != "Tüm Yıl" and secilen_ay in cluster_df.columns:
                ay_toplam = int(cluster_df[secilen_ay].sum())
                aylik_html = f"<tr><td>📅 {secilen_ay}</td><td><b style='color:#4ea8ff'>{ay_toplam}</b></td></tr>"

            popup_html = f"""
            <div style='font:13px sans-serif;min-width:220px;'>
              <div style='background:#302b63;color:#fff;padding:8px 12px;border-radius:8px 8px 0 0;font-weight:700;'>
                🚗 DURAK — Grup {cid}
              </div>
              <div style='padding:10px 12px;background:#1e2130;color:#dde;border-radius:0 0 8px 8px;'>
                <table style='width:100%;border-collapse:collapse;font-size:12px;'>
                  <tr><td>Erişim Türü</td><td><b>{stop_tur}</b></td></tr>
                  <tr><td>Trafo Sayısı</td><td><b>{len(nodes)}</b></td></tr>
                  <tr><td>Toplam Ziyaret</td><td><b>{toplam_gidilme} kez</b></td></tr>
                  {aylik_html}
                </table>
              </div>
            </div>"""

            folium.Marker(
                location=[stop_lat, stop_lon],
                icon=folium.DivIcon(
                    html="""<div style='background:#ff4757;border:3px solid #fff;border-radius:50%;
                                width:24px;height:24px;display:flex;align-items:center;
                                justify-content:center;font-size:12px;box-shadow:0 2px 8px #0006;'>
                              🚗
                            </div>""",
                    icon_size=(24, 24), icon_anchor=(12, 12)
                ),
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Grup {cid} — {len(nodes)} trafo, {toplam_gidilme} ziyaret"
            ).add_to(m)

        for idx in nodes:
            row = df.iloc[idx]
            gidilme   = int(row['GİDİLME SAYISI']) if 'GİDİLME SAYISI' in row.index else 0
            trafo_lat = float(row['CBS_Y'])
            trafo_lon = float(row['CBS_X'])
            trafo_kod = str(row['TRAFO_KODU'])
            trafo_tur = str(row['ABONE_GRUP_ADI'])

            ay_html = ""
            tooltip_extra = ""
            if secilen_ay != "Tüm Yıl" and secilen_ay in row.index:
                aylik = int(row[secilen_ay])
                renk = "#4ea8ff" if aylik > 0 else "#888"
                ay_html = f"<tr><td>📅 {secilen_ay}</td><td><b style='color:{renk}'>{aylik}</b></td></tr>"
                tooltip_extra = f" | {secilen_ay}: {aylik}"

            popup_html = f"""
            <div style='font:12px sans-serif;min-width:190px;'>
              <div style='background:{color};color:#fff;padding:7px 10px;border-radius:8px 8px 0 0;font-weight:700;'>
                ⚡ {trafo_kod}
              </div>
              <div style='padding:8px 10px;background:#1e2130;color:#ccd;border-radius:0 0 8px 8px;'>
                <table style='width:100%;border-collapse:collapse;font-size:12px;'>
                  <tr><td>Tür</td><td>{trafo_tur}</td></tr>
                  <tr><td>Toplam Ziyaret</td><td><b>{gidilme}</b></td></tr>
                  <tr><td>Grup</td><td>#{cid}</td></tr>
                  {ay_html}
                </table>
              </div>
            </div>"""

            radius = min(5 + gidilme // 5, 14)
            folium.CircleMarker(
                location=[trafo_lat, trafo_lon],
                radius=radius,
                color=color, fill=True, fill_color=color, fill_opacity=0.85,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"⚡ {trafo_kod} | Ziyaret: {gidilme}{tooltip_extra}"
            ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# ANALİZ GRAFİKLERİ
# ─────────────────────────────────────────────────────────────────────────────
def render_analytics(df, clusters):
    mevcut_aylar = [a for a in AYLAR if a in df.columns]
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Aylık Trend", "📊 Dağılım", "🏆 Top Trafolar", "🗂️ Ham Veri"])

    with tab1:
        if mevcut_aylar:
            monthly = df[mevcut_aylar].sum().reset_index()
            monthly.columns = ['Ay', 'İş Emri']
            fig = px.area(monthly, x='Ay', y='İş Emri',
                          title="Aylık İş Emri Trendi",
                          color_discrete_sequence=['#636bff'],
                          template='plotly_dark')
            fig.update_traces(fill='tozeroy', line_width=2.5, fillcolor='rgba(99,107,255,0.2)')
            fig.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                              font_family='DM Sans', height=320,
                              margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ay sütunları bulunamadı.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            grp = df['ABONE_GRUP_ADI'].value_counts().head(10).reset_index()
            grp.columns = ['Grup', 'Sayı']
            fig2 = px.bar(grp, x='Sayı', y='Grup', orientation='h',
                          title="Abone Grubu Dağılımı",
                          color='Sayı', color_continuous_scale='Purples',
                          template='plotly_dark')
            fig2.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                               height=320, showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            sizes = [len(c) for c in clusters]
            size_counts = pd.Series(sizes).value_counts().sort_index().reset_index()
            size_counts.columns = ['Boyut', 'Adet']
            fig3 = px.bar(size_counts, x='Boyut', y='Adet',
                          title="Cluster Boyut Dağılımı",
                          color='Adet', color_continuous_scale='Blues',
                          template='plotly_dark')
            fig3.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                               height=320, coloraxis_showscale=False,
                               margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        if 'GİDİLME SAYISI' in df.columns:
            top20 = df.nlargest(20, 'GİDİLME SAYISI')[
                ['TRAFO_KODU', 'ABONE_GRUP_ADI', 'MAHALLE', 'GİDİLME SAYISI']]
            fig4 = px.bar(top20, x='GİDİLME SAYISI', y='TRAFO_KODU',
                          orientation='h', color='ABONE_GRUP_ADI',
                          title="En Çok Ziyaret Edilen 20 Trafo",
                          template='plotly_dark',
                          color_discrete_sequence=px.colors.qualitative.Vivid)
            fig4.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                               height=520, margin=dict(l=20, r=20, t=50, b=20),
                               yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        display_cols = [c for c in ['TRAFO_KODU', 'ABONE_GRUP_ADI', 'İLÇE', 'MAHALLE',
                                     'GİDİLME SAYISI'] + mevcut_aylar if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values('GİDİLME SAYISI', ascending=False).reset_index(drop=True),
            use_container_width=True, height=400
        )
        st.download_button(
            "⬇️ CSV İndir",
            df[display_cols].to_csv(index=False).encode('utf-8-sig'),
            file_name="rf_analiz.csv", mime="text/csv"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>RF Drive-By — Saha Analiz Merkezi</h1>
  <p>Akıllı kümeleme • Durak optimizasyonu • İş emri analizi • RF ağ görselleştirme</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Veri Kaynağı")
    uploaded_file = st.file_uploader("Excel veya CSV yükle", type=["xlsx", "csv"],
                                     label_visibility="collapsed")

    if not uploaded_file:
        st.markdown("""
        <div class="info-box">
          👆 Başlamak için bir <b>Excel (.xlsx)</b> veya <b>CSV</b> dosyası yükleyin.<br><br>
          Beklenen sütunlar:<br>
          <code>TRAFO_KODU, CBS_X, CBS_Y, İLÇE, MAHALLE, ABONE_GRUP_ADI, GİDİLME SAYISI</code><br>
          + Ay sütunları (OCAK … ARALIK)
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    data = load_data(uploaded_file)
    if data is None:
        st.stop()

    st.markdown("---")
    st.markdown("## 📍 Konum Filtreleri")
    ilce = st.selectbox("İlçe", sorted(data['İLÇE'].unique()))
    mahalleler = sorted(data[data['İLÇE'] == ilce]['MAHALLE'].unique().tolist())
    secilen_mahalleler = st.multiselect("Mahalle (opsiyonel)", mahalleler)

    st.markdown("---")
    st.markdown("## 📅 İş Emri Filtresi")
    secilen_ay = st.selectbox("Ay seç", ["Tüm Yıl"] + AYLAR)
    sadece_is_olanlar = False
    if secilen_ay != "Tüm Yıl":
        sadece_is_olanlar = st.toggle(f"Sadece {secilen_ay} ayında işi olanlar")

    st.markdown("---")
    st.markdown("## 📶 RF Ağ Ayarları")
    rf_menzil  = st.slider("Haberleşme menzili (m)", 50, 3000, 500, 50)
    katman     = st.selectbox("Harita tipi", list(HARITA_TIPLERI.keys()))
    show_mesh  = st.toggle("RF bağlantı çizgileri", value=True)
    show_heatmap = st.toggle("Ziyaret yoğunluk haritası", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# FİLTRELEME
# ─────────────────────────────────────────────────────────────────────────────
df_f = data[data['İLÇE'] == ilce].copy()
if secilen_mahalleler:
    df_f = df_f[df_f['MAHALLE'].isin(secilen_mahalleler)]
if sadece_is_olanlar and secilen_ay in df_f.columns:
    df_f = df_f[df_f[secilen_ay] > 0]
df_f = df_f.reset_index(drop=True)

if df_f.empty:
    st.warning("⚠️ Seçilen filtrelere uygun trafo bulunamadı.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER
# ─────────────────────────────────────────────────────────────────────────────
df_serial = df_f.copy()
for col in df_serial.select_dtypes(include=[np.integer]).columns:
    df_serial[col] = df_serial[col].astype(int)
for col in df_serial.select_dtypes(include=[np.floating]).columns:
    df_serial[col] = df_serial[col].astype(float)

pairs, clusters = compute_clusters(df_serial.to_json(orient='split'), rf_menzil)
multi_nodes = len([c for c in clusters if len(c) > 1])

# ─────────────────────────────────────────────────────────────────────────────
# METRİKLER
# ─────────────────────────────────────────────────────────────────────────────
toplam_ziyaret = int(df_f['GİDİLME SAYISI'].sum()) if 'GİDİLME SAYISI' in df_f.columns else 0
ort_ziyaret    = round(toplam_ziyaret / len(df_f), 1) if len(df_f) else 0
aylik_ziyaret  = int(df_f[secilen_ay].sum()) if secilen_ay != "Tüm Yıl" and secilen_ay in df_f.columns else None

cols = st.columns(5)
cards = [
    (str(len(df_f)),        "Toplam Trafo",    "#636bff"),
    (str(len(clusters)),    "RF Grubu",        "#ff6b6b"),
    (str(multi_nodes),      "Çoklu Grup",      "#ffa502"),
    (f"{toplam_ziyaret:,}", "Toplam Ziyaret",  "#2ed573"),
    (f"{ort_ziyaret}",      "Trafo Başı Ort.", "#1e90ff"),
]
for col, (val, label, color) in zip(cols, cards):
    col.markdown(f"""
    <div class="metric-card">
      <div class="value" style="color:{color}">{val}</div>
      <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

if aylik_ziyaret is not None:
    st.markdown(f"""
    <div class="info-box" style="margin-top:.7rem;">
      📅 <b>{secilen_ay}</b> ayında toplam
      <b style="color:#4ea8ff">{aylik_ziyaret:,} iş emri</b>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HARİTA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗺️ RF Saha Haritası</div>', unsafe_allow_html=True)

with st.spinner("Harita yükleniyor..."):
    fmap = create_map(df_f, pairs, clusters, katman, secilen_ay, show_heatmap, show_mesh)
    map_key = f"map_{katman}_{rf_menzil}_{len(df_f)}_{secilen_ay}_{show_heatmap}_{show_mesh}"
    st_folium(fmap, width="100%", height=620, returned_objects=[], key=map_key)

st.caption("🔴 Durak Noktası  |  ⚫ Tekil Trafo  |  Renkli gruplar = RF kümesi")

# ─────────────────────────────────────────────────────────────────────────────
# ANALİZ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Detaylı Analiz</div>', unsafe_allow_html=True)
render_analytics(df_f, clusters)
