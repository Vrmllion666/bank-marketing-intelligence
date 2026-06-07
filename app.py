"""
app.py — Bank Marketing Intelligence Dashboard
"""
 
import os, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
warnings.filterwarnings("ignore")
 
st.set_page_config(
    page_title="Bank Marketing Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)
 
MODEL_DIR = "model"
 
NUM_COLS = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]
 
CLUSTER_LABELS = {
    0: "Cluster 0 — Mass Market",
    1: "Cluster 1 — Floating Segment",
    2: "Cluster 2 — Loyal Gold",
    3: "Cluster 3 — High Potential",
}
 
CLUSTER_DESC = {
    0: "Nasabah muda/produktif, durasi telepon pendek, konversi rendah (~29%). Cocok untuk kanal digital.",
    1: "Segmen 'ragu-ragu', sangat bergantung pada penawaran dan insentif dari tim marketing.",
    2: "Nasabah paling potensial — durasi komunikasi panjang, konversi tertinggi (~93%).",
    3: "Nasabah senior mencari stabilitas finansial, konversi tinggi (~81%). Butuh penjelasan detail.",
}
 
CLUSTER_COLORS = {0: "#2196F3", 1: "#C338B5", 2: "#4CAF50", 3: "#FF5722"}
 
@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    try:
        meta      = joblib.load(f"{MODEL_DIR}/model_metadata.pkl")
        scaler    = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        kmeans    = joblib.load(f"{MODEL_DIR}/kmeans.pkl")
        feat_cols = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
        xgb       = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
        clust_df  = pd.read_parquet(f"{MODEL_DIR}/cluster_viz.parquet")
        return meta, scaler, kmeans, feat_cols, xgb, clust_df
    except FileNotFoundError:
        return None
 
 
def check_models():
    required = ["model_metadata.pkl","scaler.pkl","kmeans.pkl",
            "feature_columns.pkl","xgboost.pkl","cluster_viz.parquet"]
    return all(os.path.exists(f"{MODEL_DIR}/{f}") for f in required)
 
 
def preprocess_input(input_dict, scaler, feat_cols):
    df = pd.DataFrame([input_dict])
    cat_mappings = {
        "job"         : ["admin.","blue-collar","entrepreneur","housemaid","management",
                         "retired","self-employed","services","student","technician","unemployed"],
        "marital"     : ["divorced","married","single"],
        "education"   : ["basic.4y","basic.6y","basic.9y","high.school","illiterate",
                         "professional.course","university.degree"],
        "default"     : ["no","yes","unknown"],
        "housing"     : ["no","yes"],
        "loan"        : ["no","yes"],
        "contact"     : ["cellular","telephone"],
        "month"       : ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
        "day_of_week" : ["mon","tue","wed","thu","fri"],
        "poutcome"    : ["failure","nonexistent","success"],
    }
    df_enc = df.copy()
    for col, categories in cat_mappings.items():
        if col in df_enc.columns:
            for cat in categories[1:]:
                df_enc[f"{col}_{cat}"] = (df_enc[col] == cat).astype(int)
            df_enc.drop(columns=[col], inplace=True)
    for c in feat_cols:
        if c not in df_enc.columns:
            df_enc[c] = 0
    df_enc = df_enc[feat_cols]
    df_enc[NUM_COLS] = scaler.transform(df_enc[NUM_COLS])
    return df_enc
 
 
# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-image:
        radial-gradient(circle at 15% 20%, rgba(55,138,221,0.06) 0%, transparent 50%),
        radial-gradient(circle at 85% 80%, rgba(76,175,80,0.04) 0%, transparent 50%);
    background-attachment: fixed;
}

/* SVG background illustration — minimalist bank/finance vectors */
.stApp::after {
    content: '';
    position: fixed;
    bottom: -60px;
    right: -60px;
    width: 560px;
    height: 560px;
    opacity: 0.055;
    pointer-events: none;
    z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Cg fill='none' stroke='%23378ADD' stroke-width='1.5'%3E%3Crect x='30' y='80' width='140' height='90' rx='4'/%3E%3Crect x='45' y='95' width='30' height='60'/%3E%3Crect x='85' y='105' width='30' height='50'/%3E%3Crect x='125' y='88' width='30' height='67'/%3E%3Cpolyline points='30,70 75,40 110,55 160,25' stroke='%234CAF50' stroke-width='2'/%3E%3Ccircle cx='75' cy='40' r='3' fill='%234CAF50'/%3E%3Ccircle cx='110' cy='55' r='3' fill='%234CAF50'/%3E%3Ccircle cx='160' cy='25' r='3' fill='%234CAF50'/%3E%3Cline x1='20' y1='170' x2='180' y2='170' stroke-width='1'/%3E%3C/g%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: contain;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 60px;
    left: -20px;
    width: 320px;
    height: 320px;
    opacity: 0.04;
    pointer-events: none;
    z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'%3E%3Cg fill='none' stroke='%23378ADD' stroke-width='1.2'%3E%3Ccircle cx='60' cy='60' r='50'/%3E%3Ccircle cx='60' cy='60' r='35'/%3E%3Ccircle cx='60' cy='60' r='20'/%3E%3Cline x1='60' y1='10' x2='60' y2='110'/%3E%3Cline x1='10' y1='60' x2='110' y2='60'/%3E%3C/g%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: contain;
}

.block-container { padding-top: 1.6rem !important; max-width: 1080px;    padding-left: 1rem !important;
    padding-right: 1rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f1e 0%, #081223 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left; background: transparent; border: none;
    border-radius: 8px; font-size: 0.88rem; color: rgba(255,255,255,0.75);
    transition: all 0.2s; padding: 8px 12px; font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(55,138,221,0.15); color: white; }
section[data-testid="stSidebar"] .stButton > button:focus { color: white; background: #2563eb; }

[data-testid="stMetric"] {
    background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px; padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: box-shadow 0.25s, border-color 0.25s;
}
[data-testid="stMetric"]:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.15); border-color: rgba(37,99,235,0.4); }
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; letter-spacing: 0.06em; color: #6b7280 !important; text-transform: uppercase; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-size: 1.85rem !important; font-weight: 800 !important; letter-spacing: -0.03em; color: var(--text-color) !important; }

.section-label {
    font-size: 10.5px; font-weight: 700; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 0 0 14px; padding-top: 2px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: rgba(128,128,128,0.2); }

.form-section-header {
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;
    color: #6b7280; margin: 4px 0 14px; padding: 10px 0 10px;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}

.result-card {
    border-radius: 16px; padding: 28px 20px; text-align: center;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 8px;
}
.result-yes { background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06)); border: 1px solid rgba(16,185,129,0.35); }
.result-no  { background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));  border: 1px solid rgba(239,68,68,0.35);  }
.result-label { font-size: 1.3rem; font-weight: 800; letter-spacing: 0.04em; margin: 0; }
.result-yes .result-label { color: #10b981; }
.result-no  .result-label { color: #ef4444; }
.result-prob { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; }
.result-prob-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; margin: 0; }

/* ── Guide box — adaptive light/dark ── */
.guide-box {
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.22);
    border-radius: 12px; padding: 16px 20px;
}
.guide-title {
    font-size: 13px; font-weight: 700; margin-bottom: 12px;
    color: #1d4ed8;  /* light mode: biru gelap, terbaca */
}
.guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px 28px; }
.guide-item {
    font-size: 12px;
    color: var(--text-color);   /* ikut tema Streamlit otomatis */
    line-height: 1.5; padding: 4px 0;
    border-bottom: 1px solid rgba(128,128,128,0.15);
}
.guide-item strong {
    color: var(--text-color);   /* ikut tema */
    font-weight: 700;
}

/* ── Cluster card desc text — adaptive ── */
.cluster-desc-adaptive {
    font-size: 11.5px;
    color: var(--text-color);
    opacity: 0.7;
    margin: 0 0 14px;
    line-height: 1.6;
}

/* ── Streamlit dark: guide-title biru terang ── */
html[data-theme="dark"] .guide-title { color: #60a5fa !important; }
html[data-theme="light"] .guide-title { color: #1d4ed8 !important; }

[data-testid="stExpander"] { border: 1px solid rgba(55,138,221,0.2) !important; border-radius: 12px !important; }

.sidebar-info { font-size: 0.8rem; color: rgba(255,255,255,0.4); line-height: 1.6; }
[data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label { font-size: 0.8rem !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        line-height: 1.2;
        padding: 1rem 0.5rem;
    ">
        <span style="
            font-size: clamp(16px, 2.5vw, 32px);
            font-weight: bold;
            color: white;
            margin: 0;
            padding: 0;
            word-break: break-word;
        ">Bank Marketing</span>
        <span style="
            font-size: clamp(13px, 2vw, 25px);
            font-weight: bold;
            color: white;
            margin: 0;
            padding: 0;
            word-break: break-word;
        ">Intelligence</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
        <hr style="
    margin: 20px 0px;
    border: none;
    border-top: 0px;">
        """, unsafe_allow_html=True)
 
    if not check_models():
        st.error("Model belum tersedia!")
        st.code("python train_model.py", language="bash")
        st.stop()
 
    artifacts = load_artifacts()
    if artifacts is None:
        st.error("Gagal memuat model.")
        st.stop()
 
    meta, scaler, kmeans, feat_cols, xgb_model, clust_df = artifacts
 
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    
 
    # Navigasi tombol — tanpa radio button
    menu_items = [
        "Home",
        "Dataset",
        "Dashboard",
        "Prediksi Nasabah",
        "Prediksi Batch (CSV)",
        "Evaluasi Model",
        "About"
    ]
 
 
    for label in menu_items:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
 
    page = st.session_state.page
 
    st.markdown("""
        <hr style="
    margin: 0px 0px;
    border: none;
    border-top: 1px solid rgba(1,1,1, 0.2);
    padding-bottom:50px;
    color:white;
        ">
        """, unsafe_allow_html=True)
    
 
 
# ═══════════════════════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI NASABAH
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# HALAMAN 3 — SEGMENTASI CLUSTER (DASHBOARD)
# ═══════════════════════════════════════════════════════════════
 
@st.cache_data
def load_original_stats(_scaler, _feat_cols, _kmeans):
    # Load cluster assignment dari parquet (konsisten dengan notebook/SMOTE)
    df_smote = pd.read_parquet(os.path.join(MODEL_DIR, "data_resampled.parquet"))

    # Hitung konversi per cluster dari data SMOTE (konsisten dengan notebook)
    conv_smote = df_smote.groupby("cluster")["y_bin"].mean() * 100

    # Load data asli hanya untuk avg_konversi dan total_nasabah yang benar
    df_orig = pd.read_csv("bank-additional-full.csv", sep=";")
    df_orig["y_bin"] = (df_orig["y"] == "yes").astype(int)
    # Simpan avg konversi dari data asli sebagai atribut tambahan
    df_orig.attrs["avg_konversi_asli"] = df_orig["y_bin"].mean() * 100
    df_orig.attrs["total_nasabah_asli"] = len(df_orig)

    return df_orig, conv_smote, df_smote["cluster"].value_counts().sort_index()

if page == "Home":
    st.markdown("""
    <div style='padding:4px 0 24px;'>
        <p style='font-size:2.2rem;font-weight:800;letter-spacing:-0.03em;margin:0 0 8px;'>
            Bank Marketing Intelligence
        </p>
        <p style='font-size:1rem;color:var(--text-color);opacity:0.6;margin:0 0 32px;'>
            Sistem prediksi dan segmentasi nasabah berbasis Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-label'style= color:black;>Deskripsi Proyek</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.22);
                border-radius:12px;padding:20px 24px;margin-bottom:24px;'>
        <p style='font-size:0.95rem;line-height:1.8;margin:0;color:var(--text-color);'>
            Proyek ini membangun sistem <strong>Bank Marketing Intelligence</strong> menggunakan 
            dataset UCI Bank Marketing. Model machine learning digunakan untuk memprediksi 
            kemungkinan nasabah berlangganan deposito berjangka, serta mengelompokkan nasabah 
            ke dalam segmen menggunakan K-Means Clustering untuk mendukung strategi pemasaran 
            yang lebih tepat sasaran.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Identitas Anggota</p>", unsafe_allow_html=True)
    members = [
        ("Weka Surajati Sudanta", "NIM: 24051214079" ,"https://raw.githubusercontent.com/Vrmllion666/bank-marketing-intelligence/main/image/agr.jpg"),
        ("Agril Adirizky", "NIM: 24051214095","https://raw.githubusercontent.com/Vrmllion666/bank-marketing-intelligence/main/image/wk.jpeg"),
    ]
    cols = st.columns(len(members))
    for col, (nama, nim, fotoprofil) in zip(cols, members):
        with col:
            st.markdown(f"""
            <div style='background:rgba(128,128,128,0.07);border:1px solid rgba(128,128,128,0.15);
                        border-radius:12px;padding:20px;text-align:center;'>
                <img src='{fotoprofil}' style='
                width:72px;
                height:72px;
                border-radius:50%;
                object-fit:cover;
                display:block;
                margin:0 auto 12px;
                border:2px solid rgba(37,99,235,0.3);'>
                <p style='font-size:14px;font-weight:700;margin:0 0 4px;'>{nama}</p>
                <p style='font-size:12px;color:#6b7280;margin:0 0 2px;'>{nim}</p>
            </div>
            """, unsafe_allow_html=True)
    
elif page == "Dataset":
    st.markdown("""
    <div style='padding:4px 0 16px;'>
        <p style='font-size:2rem;font-weight:800;letter-spacing:-0.03em;margin:0 0 4px;'>Dataset Overview</p>
        <p style='font-size:0.87rem;color:var(--text-color);opacity:0.6;margin:0;'>
            Informasi dan statistik dataset Bank Marketing (UCI ML Repository)
        </p>
    </div>
    """, unsafe_allow_html=True)

    df_info = pd.read_csv("bank-additional-full.csv", sep=";")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{len(df_info):,}")
    c2.metric("Jumlah Fitur", f"{len(df_info.columns)-1}")
    c3.metric("Subscribe (Yes)", f"{(df_info['y']=='yes').sum():,}")
    c4.metric("Subscribe (No)", f"{(df_info['y']=='no').sum():,}")

    st.divider()

    st.markdown("<p class='section-label'>Informasi Dataset</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.15);
                border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
        <p style='font-size:13px;line-height:1.8;margin:0;color:var(--text-color);'>
            <strong>Sumber:</strong> UCI Machine Learning Repository — Bank Marketing Dataset<br>
            <strong>Konteks:</strong> Data kampanye telemarketing bank Portugal (2008–2013)<br>
            <strong>Target:</strong> Apakah nasabah berlangganan deposito berjangka? (yes/no)<br>
            <strong>Tipe:</strong> Binary Classification + Clustering
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Statistik Deskriptif</p>", unsafe_allow_html=True)
    st.dataframe(df_info[NUM_COLS].describe().round(2), use_container_width=True)

    st.divider()

    st.markdown("<p class='section-label'>Visualisasi Data</p>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        fig_target = px.pie(df_info, names="y", title="Distribusi Target (Subscribe)",
                            color_discrete_sequence=["#2196F3","#4CAF50"])
        fig_target.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_target, use_container_width=True)
    with v2:
        fig_age = px.histogram(df_info, x="age", color="y", title="Distribusi Usia Nasabah",
                               color_discrete_sequence=["#2196F3","#4CAF50"], nbins=30)
        fig_age.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_age, use_container_width=True)

    v3, v4 = st.columns(2)
    with v3:
        fig_job = px.histogram(df_info, x="job", color="y", title="Subscribe per Pekerjaan",
                               color_discrete_sequence=["#2196F3","#4CAF50"])
        fig_job.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20), xaxis_tickangle=-30)
        st.plotly_chart(fig_job, use_container_width=True)
    with v4:
        fig_dur = px.box(df_info, x="y", y="duration", color="y",
                         title="Durasi Telepon vs Subscribe",
                         color_discrete_sequence=["#2196F3","#4CAF50"])
        fig_dur.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_dur, use_container_width=True)

    st.divider()
    st.markdown("<p class='section-label'>Sample Data (10 Baris Pertama)</p>", unsafe_allow_html=True)
    st.dataframe(df_info.head(10), use_container_width=True)

elif page == "Prediksi Nasabah":
    st.markdown("""
    <div style='padding:4px 0 16px;'>
        <p style='font-size:2rem;font-weight:800;letter-spacing:-0.03em;margin:0 0 4px;'>Prediksi Subscription Deposito</p>
        <p style='font-size:0.87rem;color:var(--text-color);opacity:0.6;margin:0;'>Masukkan data nasabah untuk memprediksi kemungkinan berlangganan deposito berjangka.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Panduan Pengisian Fitur — klik untuk melihat penjelasan tiap fitur", expanded=False):
        st.markdown("""
        <div class='guide-box'>
            <p class='guide-title'>Penjelasan Fitur Input</p>
            <div class='guide-grid'>
                <div class='guide-item'><strong>Usia</strong> — Usia nasabah dalam tahun</div>
                <div class='guide-item'><strong>Pekerjaan</strong> — Jenis pekerjaan nasabah</div>
                <div class='guide-item'><strong>Status Pernikahan</strong> — married / single / divorced</div>
                <div class='guide-item'><strong>Pendidikan</strong> — Tingkat pendidikan terakhir</div>
                <div class='guide-item'><strong>Kredit Bermasalah?</strong> — Nasabah memiliki kredit default</div>
                <div class='guide-item'><strong>Punya KPR?</strong> — Kredit kepemilikan rumah</div>
                <div class='guide-item'><strong>Punya Pinjaman?</strong> — Pinjaman personal selain KPR</div>
                <div class='guide-item'><strong>Tipe Kontak</strong> — cellular = HP, telephone = rumah</div>
                <div class='guide-item'><strong>Bulan & Hari</strong> — Waktu terakhir nasabah dihubungi</div>
                <div class='guide-item'><strong>Durasi Telepon</strong> — Makin lama = peluang subscribe lebih tinggi</div>
                <div class='guide-item'><strong>Jumlah Kontak</strong> — Berapa kali dihubungi kampanye ini</div>
                <div class='guide-item'><strong>Pdays = 999</strong> — Belum pernah dihubungi sebelumnya</div>
                <div class='guide-item'><strong>Hasil Kampanye Lalu</strong> — success / failure / nonexistent</div>
                <div class='guide-item'><strong>Euribor3M</strong> — Suku bunga pasar (rendah = kondusif)</div>
                <div class='guide-item'><strong>Emp. Variation Rate</strong> — Indikator kondisi tenaga kerja</div>
                <div class='guide-item'><strong>Nr. Employed</strong> — Jumlah karyawan aktif (ekonomi)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1], gap="large")
 
    with col_left:
        st.markdown("<p class='form-section-header'>Data Profil Nasabah</p>", unsafe_allow_html=True)
 
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Usia", 18, 95, 35)
            job = st.selectbox("Pekerjaan", [
                "admin.","blue-collar","entrepreneur","housemaid","management",
                "retired","self-employed","services","student","technician","unemployed"
            ])
            marital = st.selectbox("Status Pernikahan", ["married","single","divorced"])
 
        with c2:
            education = st.selectbox("Pendidikan", [
                "university.degree","high.school","professional.course",
                "basic.9y","basic.6y","basic.4y","illiterate"
            ])
            default  = st.selectbox("Kredit Bermasalah?", ["no","yes","unknown"])
            housing  = st.selectbox("Punya KPR?", ["yes","no"])
 
        with c3:
            loan     = st.selectbox("Punya Pinjaman?", ["no","yes"])
            contact  = st.selectbox("Tipe Kontak", ["cellular","telephone"])
            month    = st.selectbox("Bulan Kontak", [
                "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
            ], index=4)
 
        st.divider()
        st.markdown("<p class='form-section-header' style='margin-top:8px;'>Data Kampanye</p>", unsafe_allow_html=True)
 
        c4, c5, c6 = st.columns(3)
        with c4:
            day_of_week = st.selectbox("Hari", ["mon","tue","wed","thu","fri"])
            duration    = st.number_input("Durasi Telepon (detik)", 0, 5000, 200)
            campaign    = st.number_input("Jumlah Kontak", 1, 60, 2)
 
        with c5:
            pdays    = st.number_input("Pdays (999=belum)", 0, 999, 999)
            previous = st.number_input("Kontak Sebelumnya", 0, 10, 0)
            poutcome = st.selectbox("Hasil Kampanye Lalu", ["nonexistent","failure","success"])
 
        with c6:
            emp_var_rate   = st.number_input("Emp. Var. Rate", -4.0, 2.0, 1.1, 0.1)
            cons_price_idx = st.number_input("Cons. Price Idx", 92.0, 95.0, 93.5, 0.1)
            cons_conf_idx  = st.number_input("Cons. Conf. Idx", -55.0, -25.0, -36.4, 0.1)
 
        c7, c8 = st.columns(2)
        with c7:
            euribor3m   = st.number_input("Euribor3M", 0.0, 6.0, 4.8, 0.1)
        with c8:
            nr_employed = st.number_input("Nr. Employed", 4900.0, 5300.0, 5191.0, 1.0)
 
    with col_right:
        st.markdown("<p class='form-section-header'>Hasil Prediksi</p>", unsafe_allow_html=True)
 
        input_dict = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "duration": duration, "campaign": campaign, "pdays": pdays,
            "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "nr.employed": nr_employed
        }
 
        if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
            with st.spinner("Menganalisis..."):
                X_input = preprocess_input(input_dict, scaler, feat_cols)
                pred    = xgb_model.predict(X_input)[0]
                prob    = xgb_model.predict_proba(X_input)[0]
                cluster = kmeans.predict(X_input)[0]
 
            st.divider()
 
            if pred == 1:
                st.markdown(f"""
                <div class='result-card result-yes'>
                    <p class='result-label'>AKAN SUBSCRIBE</p>
                    <p class='result-prob'>{prob[1]*100:.1f}%</p>
                    <p class='result-prob-label'>Probabilitas Subscribe</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-card result-no'>
                    <p class='result-label'>TIDAK SUBSCRIBE</p>
                    <p class='result-prob'>{prob[0]*100:.1f}%</p>
                    <p class='result-prob-label'>Probabilitas Tidak Subscribe</p>
                </div>""", unsafe_allow_html=True)
 
            st.divider()
 
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[1]*100,
                title={"text": "Probabilitas Subscribe (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#059669" if pred == 1 else "#dc2626"},
                    "steps": [
                        {"range": [0, 40],  "color": "#fee2e2"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100],"color": "#d1fae5"},
                    ],
                    "threshold": {"line": {"color": "black","width": 3},
                                  "thickness": 0.75, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
 
            st.info(f"**Segmen Nasabah:**\n\n{CLUSTER_LABELS[cluster]}\n\n{CLUSTER_DESC[cluster]}")
 
            st.subheader("Rekomendasi")
            if pred == 1 and prob[1] > 0.7:
                st.success("**Prioritas Tinggi** — Segera follow up dan berikan penawaran eksklusif.")
            elif pred == 1:
                st.warning("**Potensi Sedang** — Lanjutkan komunikasi, tawarkan benefit tambahan.")
            elif cluster in [1]:
                st.warning("**Masih Bisa Dipengaruhi** — Coba tawarkan insentif atau promo bunga spesial.")
            else:
                st.error("**Prioritas Rendah** — Pertimbangkan kanal komunikasi digital (SMS/Email).")
 
        else:
            st.markdown("""
            <div style='background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.22);
                        border-radius:12px;padding:16px 18px;margin-bottom:10px;'>
                <p style='font-size:11px;font-weight:700;color:#1d4ed8;margin:0 0 12px;
                           text-transform:uppercase;letter-spacing:0.07em;' class='guide-adaptive-title'>
                    Panduan Pengisian
                </p>
                <div style='display:flex;flex-direction:column;gap:7px;'>
                    <div style='font-size:12px;padding:6px 10px;background:rgba(128,128,128,0.07);border-radius:6px;color:var(--text-color);'>
                        <strong>Durasi Telepon</strong><br>
                        <span style='opacity:0.7;'>Semakin lama = probabilitas subscribe makin tinggi</span>
                    </div>
                    <div style='font-size:12px;padding:6px 10px;background:rgba(128,128,128,0.07);border-radius:6px;color:var(--text-color);'>
                        <strong>Pdays = 999</strong><br>
                        <span style='opacity:0.7;'>Nasabah belum pernah dihubungi sebelumnya</span>
                    </div>
                    <div style='font-size:12px;padding:6px 10px;background:rgba(128,128,128,0.07);border-radius:6px;color:var(--text-color);'>
                        <strong>Euribor 3M rendah</strong><br>
                        <span style='opacity:0.7;'>Suku bunga rendah = kondusif untuk deposito</span>
                    </div>
                    <div style='font-size:12px;padding:6px 10px;background:rgba(128,128,128,0.07);border-radius:6px;color:var(--text-color);'>
                        <strong>Poutcome = success</strong><br>
                        <span style='opacity:0.7;'>Kampanye lalu berhasil = sinyal kuat subscribe</span>
                    </div>
                </div>
            </div>
            <div style='background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.15);
                        border-radius:10px;padding:12px 16px;text-align:center;'>
                <p style='font-size:12px;color:var(--text-color);opacity:0.6;margin:0;'>
                    Isi semua data lalu klik<br>
                    <strong style='color:#2563eb;opacity:1;'>Prediksi Sekarang</strong>
                </p>
            </div>
            <style>
                html[data-theme="dark"] .guide-adaptive-title { color: #60a5fa !important; }
            </style>
            """, unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI BATCH
# ═══════════════════════════════════════════════════════════════
elif page == "Prediksi Batch (CSV)":
    st.markdown("""
    <div style='padding:4px 0 16px;'>
        <p style='font-size:2rem;font-weight:800;letter-spacing:-0.03em;margin:0 0 4px;'>Prediksi Batch dari File CSV</p>
        <p style='font-size:0.87rem;color:var(--text-color);opacity:0.6;margin:0;'>Upload file CSV berisi data banyak nasabah sekaligus untuk diprediksi massal.</p>
    </div>
    """, unsafe_allow_html=True)
 
    template_cols = [
        "age","job","marital","education","default","housing","loan",
        "contact","month","day_of_week","duration","campaign","pdays",
        "previous","poutcome","emp.var.rate","cons.price.idx",
        "cons.conf.idx","euribor3m","nr.employed"
    ]
    template_data = {
        "age":[35,52,28],"job":["admin.","retired","technician"],
        "marital":["married","divorced","single"],
        "education":["university.degree","high.school","basic.9y"],
        "default":["no","no","unknown"],"housing":["yes","no","yes"],
        "loan":["no","no","yes"],"contact":["cellular","telephone","cellular"],
        "month":["may","jul","oct"],"day_of_week":["mon","wed","fri"],
        "duration":[200,420,90],"campaign":[2,1,5],
        "pdays":[999,10,999],"previous":[0,1,0],
        "poutcome":["nonexistent","success","failure"],
        "emp.var.rate":[1.1,-1.8,1.4],"cons.price.idx":[93.5,93.1,94.5],
        "cons.conf.idx":[-36.4,-41.8,-50.0],"euribor3m":[4.8,0.9,4.2],
        "nr.employed":[5191.0,5099.1,5195.8]
    }
    template_df  = pd.DataFrame(template_data)
    csv_template = template_df.to_csv(index=False).encode("utf-8")
 
    col_dl, _ = st.columns([1, 2])
    with col_dl:
        st.download_button("Download Template CSV", data=csv_template,
                           file_name="template_nasabah.csv", mime="text/csv")
 
    uploaded = st.file_uploader("Upload file CSV nasabah", type=["csv"])
 
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"File berhasil dimuat: **{len(df_up):,} baris**, {len(df_up.columns)} kolom")
 
        missing_cols = [c for c in template_cols if c not in df_up.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan: `{missing_cols}`")
            st.stop()
 
        with st.spinner("Memproses prediksi..."):
            results = []
            for _, row in df_up.iterrows():
                input_dict = row[template_cols].to_dict()
                X_in  = preprocess_input(input_dict, scaler, feat_cols)
                pred  = xgb_model.predict(X_in)[0]
                prob  = xgb_model.predict_proba(X_in)[0][1]
                clust = kmeans.predict(X_in)[0]
                results.append({
                    "Prediksi"         : "Subscribe" if pred == 1 else "Tidak Subscribe",
                    "Probabilitas (%)" : round(prob * 100, 2),
                    "Segmen"           : CLUSTER_LABELS[clust],
                })
 
        result_df = pd.concat([df_up.reset_index(drop=True), pd.DataFrame(results)], axis=1)
 
        total   = len(result_df)
        n_yes   = (result_df["Prediksi"] == "Subscribe").sum()
        n_no    = total - n_yes
        pct_yes = n_yes / total * 100
 
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Nasabah", f"{total:,}")
        c2.metric("Prediksi Subscribe", f"{n_yes:,}", f"{pct_yes:.1f}%")
        c3.metric("Prediksi Tidak", f"{n_no:,}", f"{100-pct_yes:.1f}%")
 
        seg_counts = result_df["Segmen"].value_counts().reset_index()
        seg_counts.columns = ["Segmen","Jumlah"]
        fig_seg = px.bar(seg_counts, x="Segmen", y="Jumlah",
                         title="Distribusi Nasabah per Segmen", color="Segmen",
                         color_discrete_sequence=list(CLUSTER_COLORS.values()))
        fig_seg.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_seg, use_container_width=True)
 
        st.dataframe(result_df, use_container_width=True)
 
        csv_result = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Hasil Prediksi", data=csv_result,
                           file_name="hasil_prediksi.csv", mime="text/csv")
    else:
        st.info("Upload file CSV untuk memulai prediksi massal.")
 
 
# ═══════════════════════════════════════════════════════════════
# HALAMAN 3 — SEGMENTASI CLUSTER (DASHBOARD)
# ═══════════════════════════════════════════════════════════════
elif page == "Dashboard":
 
    df_orig, conv_per_cluster, size_per_cluster = load_original_stats(scaler, feat_cols, kmeans)
    total_nasabah = 41188
    avg_konversi  = 11.3
    best_cluster     = int(conv_per_cluster.idxmax())
    CLUSTER_CONV     = {i: round(conv_per_cluster[i], 1) for i in range(4)}
    CLUSTER_COLORS_HEX = {
        0: "#378ADD",
        1: "#FFC107",
        2: "#4CAF50",
        3: "#E80C00",
    }
 
    # ── Header ───────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:left; padding:8px 0 22px;'>
        <p style='font-size:2.2rem; font-weight:800; letter-spacing:-0.03em; margin:0 0 6px;'>Dashboard</p>
        <p style='font-size:0.87rem;color:var(--text-color);opacity:0.6;margin:0;'>Ringkasan segmentasi nasabah Bank Marketing &mdash; K-Means Clustering (K=4)</p>
    </div>
    """, unsafe_allow_html=True)
 
    
    
    # ── Metric Cards ─────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric( "Total Nasabah",     f"{total_nasabah:,}",   )
    m2.metric("Jumlah Cluster",    "4",             )
    m3.metric("Rata-rata Konversi",f"{avg_konversi:.1f}%",)
    m4.metric("Cluster Terbaik",   f"Cluster {best_cluster}", )
    
    # ── Profil Setiap Segmen ──────────────────────────────────────
    st.markdown("<p class='section-label'>Profil Setiap Segmen</p>", unsafe_allow_html=True)
 
    c0, c1, c2, c3 = st.columns(4)
    for col, i in zip([c0, c1, c2, c3], range(4)):
        color = CLUSTER_COLORS_HEX[i]
        pct   = CLUSTER_CONV[i]
        with col:
            st.markdown(f"""
            <div style='
                background:linear-gradient(145deg,rgba(255,255,255,0.025),rgba(255,255,255,0.01));
                border:1px solid rgba(255,255,255,0.07);
                border-left:4px solid {color};
                border-radius:14px; padding:16px 18px; margin-bottom:6px; min-height:175px;
                box-shadow:0 2px 12px rgba(0,0,0,0.12);'>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <div style="width:9px;height:9px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                    <p style="font-size:13px;font-weight:700;margin:0;color:{color};">{CLUSTER_LABELS[i]}</p>
                </div>
                <p style="font-size:11.5px;color:var(--text-color);opacity:0.65;margin:0 0 14px;line-height:1.6;">{CLUSTER_DESC[i]}</p>
                <div style="background:rgba(128,128,128,0.15);border-radius:4px;height:5px;margin-bottom:7px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color}99,{color});border-radius:4px;"></div>
                </div>
                <p style="font-size:11px;color:var(--text-color);opacity:0.55;text-align:right;margin:0;font-weight:600;">
                    Konversi: <span style="color:{color};font-size:13px;opacity:1;">{pct}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
 
 
 
    #    Chart Row  
    chart_left, chart_right = st.columns(2)
 
    with chart_left:
        conv_df = pd.DataFrame({
            "Cluster": [CLUSTER_LABELS[i] for i in range(4)],
            "Konversi (%)": [CLUSTER_CONV[i] for i in range(4)],
            "Warna": [CLUSTER_COLORS_HEX[i] for i in range(4)],
        })
        fig_conv = go.Figure()
        for _, row in conv_df.iterrows():
            fig_conv.add_trace(go.Bar(
                x=[row["Cluster"]], y=[row["Konversi (%)"]],
                marker_color=row["Warna"],
                text=[f"{row['Konversi (%)']:.1f}%"],
                textposition="outside",
                showlegend=False,
            ))
        fig_conv.update_layout(
            title="Tingkat konversi per cluster",
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis_range=[0, 115],
            yaxis_title="Konversi (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_conv.update_xaxes(tickfont=dict(size=11))
        st.plotly_chart(fig_conv, use_container_width=True)
 
    with chart_right:
        size_df = pd.DataFrame({
            "Cluster": [CLUSTER_LABELS[i] for i in range(4)],
            "Jumlah" : [int(size_per_cluster.get(i, 0)) for i in range(4)],
            "Warna"  : [CLUSTER_COLORS_HEX[i] for i in range(4)],
        })
        fig_pie = go.Figure(go.Pie(
            labels=size_df["Cluster"],
            values=size_df["Jumlah"],
            marker_colors=size_df["Warna"].tolist(),
            hole=0.45,
            textinfo="percent",
            hovertemplate="%{label}<br>%{value:,} nasabah<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            title="Proporsi nasabah per segmen",
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(font=dict(size=11), orientation="v"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
 
    #  PCA Scatter Plot 
    st.markdown("<p class='section-label'>Visualisasi Cluster (PCA 2D)</p>", unsafe_allow_html=True)
 
    sample_size = min(15000, len(clust_df))
    df_sample   = clust_df.sample(n=sample_size, random_state=42).copy()
    df_sample["Segmen"]    = df_sample["cluster"].map(CLUSTER_LABELS)
    df_sample["Subscribe"] = df_sample["y_bin"].map({1: "Ya", 0: "Tidak"})
 
    fig_scatter = px.scatter(
    df_sample,
    x="PC1",
    y="PC2",
    color="Segmen",
    symbol="Subscribe",
    opacity=0.45,
    color_discrete_map={v: CLUSTER_COLORS_HEX[k] for k, v in CLUSTER_LABELS.items()},
    labels={
        "PC1": "Principal Component 1",
        "PC2": "Principal Component 2"
    },
)
    
    fig_scatter.update_traces(marker=dict(size=3))
    fig_scatter.update_layout(
    height=420,
    margin=dict(t=20, b=120, l=20, r=20),
    legend=dict(
        font=dict(size=13),
        orientation="h",
        y=-0.28,
        x=0,
        itemsizing="constant",
        itemwidth=40,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
 
    st.divider()
 
    #  Rekomendasi Strategi 
    st.markdown("<p class='section-label'>Rekomendasi Strategi</p>", unsafe_allow_html=True)
 
    recs = [
        ("ti-target-arrow", "Cluster 2 & 3 — Fokus sumber daya sales",
         "Alokasikan porsi terbesar tim ke segmen ini. ROI tertinggi."),
        ("ti-gift",         "Cluster 1 — Pendekatan insentif",
         "Tawarkan bunga spesial atau hadiah untuk mendorong keputusan."),
        ("ti-device-mobile","Cluster 0 — Kanal digital",
         "Ganti telepon dengan SMS blast atau email untuk efisiensi biaya."),
        ("ti-refresh",      "Pemantauan berkala",
         "Re-train dan re-cluster setiap kuartal sesuai kondisi ekonomi makro."),
    ]
 
    r1, r2 = st.columns(2)
    for idx, (icon, title, body) in enumerate(recs):
        col = r1 if idx % 2 == 0 else r2
        with col:
            st.markdown(f"""
            <div style='display:flex; gap:12px; align-items:flex-start;
                        background:rgba(128,128,128,0.07);
                        border-radius:10px; padding:14px; margin-bottom:10px;'>
                <div style='font-size:18px; color:#888; margin-top:1px; flex-shrink:0;'>
                    <i class="ti {icon}"></i>
                </div>
                <div>
                    <p style='font-size:13px; font-weight:500; margin:0 0 4px;
                              color:var(--text-color);'>{title}</p>
                    <p style='font-size:12px; color:#888; margin:0; line-height:1.5;'>{body}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
 

# HALAMAN 4 — EVALUASI MODEL

elif page == "Evaluasi Model":
    st.markdown("""
    <div style='text-align:center; padding:8px 0 18px;'>
        <p style='font-size:2rem;font-weight:800;letter-spacing:-0.03em;margin:0 0 4px;'>Evaluasi Performa Model</p>
        <p style='font-size:0.87rem;color:var(--text-color);opacity:0.6;margin:0;'>Perbandingan Random Forest vs XGBoost pada data uji (hold-out 20%)</p>
    </div>
    """, unsafe_allow_html=True)
 
    rf_m  = meta["rf_metrics"]
    xgb_m = meta["xgb_metrics"]
    best  = meta["best_model_name"]
 
    st.subheader(f"Model Terbaik: {best}")
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_display = [
        ("Accuracy",  xgb_m["Accuracy"]),
        ("Precision", xgb_m["Precision"]),
        ("Recall",    xgb_m["Recall"]),
        ("F1-Score",  xgb_m["F1-Score"]),
        ("ROC-AUC",   xgb_m["ROC-AUC"]),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5], metrics_display):
        col.metric(label, f"{val:.4f}")
 
    st.divider()
 
    st.subheader("Perbandingan Metrik: RF vs XGBoost")
    metrics_keys = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    comp_df = pd.DataFrame({
        "Metrik"       : metrics_keys,
        "Random Forest": [rf_m[k] for k in metrics_keys],
        "XGBoost"      : [xgb_m[k] for k in metrics_keys],
    })
    st.dataframe(
        comp_df.style
            .highlight_max(subset=["Random Forest","XGBoost"], axis=1, color="#6fc0d7")
            .format({"Random Forest": "{:.4f}", "XGBoost": "{:.4f}"}),
        use_container_width=True
    )
 
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Random Forest", x=metrics_keys,
        y=[rf_m[k] for k in metrics_keys],
        marker_color="#2196F3", text=[f"{rf_m[k]:.3f}" for k in metrics_keys],
        textposition="outside"
    ))
    fig_comp.add_trace(go.Bar(
        name="XGBoost", x=metrics_keys,
        y=[xgb_m[k] for k in metrics_keys],
        marker_color="#FF5722", text=[f"{xgb_m[k]:.3f}" for k in metrics_keys],
        textposition="outside"
    ))
    fig_comp.update_layout(
        barmode="group", height=400,
        title="Perbandingan Metrik Evaluasi: Random Forest vs XGBoost",
        yaxis_range=[0, 1.15],
        legend=dict(orientation="h", y=1.1),
        shapes=[dict(type="line", x0=-0.5, x1=4.5, y0=0.8, y1=0.8,
                     line=dict(color="red", width=1.5, dash="dash"))]
    )
    st.plotly_chart(fig_comp, use_container_width=True)
 
    st.divider()
    st.subheader("Tentang Model & Pipeline")
    with st.expander("Detail Pipeline Training"):
        st.markdown(f"""
        **Dataset:** Bank Marketing (UCI ML Repository) — {meta['n_features']} fitur setelah encoding
 
        **Pipeline:**
        1. **Cleaning** — hapus duplikat, hapus baris dengan nilai 'unknown' pada fitur kritis
        2. **Encoding** — One-Hot Encoding untuk fitur kategorikal (drop_first=True)
        3. **Scaling** — StandardScaler pada {len(NUM_COLS)} fitur numerik
        4. **Balancing** — SMOTE untuk mengatasi class imbalance
        5. **Clustering** — KMeans K=4
        6. **Classification** — Random Forest (300 trees) vs XGBoost (300 rounds, lr=0.05)
        7. **Evaluasi** — Train/Test split 80:20 dengan stratified sampling
 
        **Metrik Utama:** ROC-AUC (threshold-independent, cocok untuk imbalanced data)
        """)
 
    with st.expander("Fitur Paling Berpengaruh"):
        st.markdown("""
        - **duration** — Durasi telepon (prediktor terkuat)
        - **euribor3m** — Suku bunga pasar
        - **nr.employed** — Jumlah tenaga kerja
        - **emp.var.rate** — Employment variation rate
        - **pdays** — Hari sejak kontak terakhir
        - **poutcome_success** — Hasil kampanye sebelumnya berhasil
        """)

elif page == "About":
    st.markdown("""
    <div style='padding:4px 0 16px;'>
        <p style='font-size:2rem;font-weight:800;letter-spacing:-0.03em;margin:0 0 4px;'>About</p>
        <p style='font-size:0.87rem;color:var(--text-color);opacity:0.6;margin:0;'>
            Penjelasan metode, dataset, dan informasi proyek
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Penjelasan Metode</p>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:rgba(128,128,128,0.07);border:1px solid rgba(128,128,128,0.15);
                    border-radius:12px;padding:18px 20px;margin-bottom:12px;'>
            <p style='font-size:13px;font-weight:700;margin:0 0 8px;'>XGBoost</p>
            <p style='font-size:12px;color:#6b7280;margin:0;line-height:1.7;'>
                Algoritma boosting berbasis decision tree. Digunakan untuk klasifikasi biner 
                apakah nasabah akan subscribe deposito atau tidak. Dipilih karena performa 
                tinggi pada data tabular dan imbalanced dataset.
            </p>
        </div>
        <div style='background:rgba(128,128,128,0.07);border:1px solid rgba(128,128,128,0.15);
                    border-radius:12px;padding:18px 20px;'>
            <p style='font-size:13px;font-weight:700;margin:0 0 8px;'>Random Forest</p>
            <p style='font-size:12px;color:#6b7280;margin:0;line-height:1.7;'>
                Ensemble learning berbasis banyak decision tree. Digunakan sebagai model 
                pembanding terhadap XGBoost untuk memvalidasi performa prediksi.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:rgba(128,128,128,0.07);border:1px solid rgba(128,128,128,0.15);
                    border-radius:12px;padding:18px 20px;margin-bottom:12px;'>
            <p style='font-size:13px;font-weight:700;margin:0 0 8px;'>K-Means Clustering</p>
            <p style='font-size:12px;color:#6b7280;margin:0;line-height:1.7;'>
                Algoritma unsupervised learning untuk mengelompokkan nasabah ke dalam 4 segmen 
                berdasarkan karakteristik perilaku dan demografi mereka.
            </p>
        </div>
        <div style='background:rgba(128,128,128,0.07);border:1px solid rgba(128,128,128,0.15);
                    border-radius:12px;padding:18px 20px;'>
            <p style='font-size:13px;font-weight:700;margin:0 0 8px;'>SMOTE</p>
            <p style='font-size:12px;color:#6b7280;margin:0;line-height:1.7;'>
                Synthetic Minority Oversampling Technique. Digunakan untuk mengatasi 
                ketidakseimbangan kelas pada data training (hanya ~11% nasabah yang subscribe).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<p class='section-label'>Dataset</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.22);
                border-radius:12px;padding:20px 24px;margin-bottom:24px;'>
        <p style='font-size:13px;line-height:1.8;margin:0;color:var(--text-color);'>
            <strong>Nama Dataset:</strong> Bank Marketing Dataset<br>
            <strong>Sumber:</strong> UCI Machine Learning Repository<br>
            <strong>Periode:</strong> 2008 – 2013 (Bank Portugal)<br>
            <strong>Jumlah Data:</strong> 41,188 baris, 20 fitur input + 1 target<br>
            <strong>Target:</strong> Variabel <code>y</code> — apakah nasabah subscribe deposito berjangka (yes/no)<br>
            <strong>Tipe Fitur:</strong> Numerik (10 fitur) dan Kategorikal (10 fitur)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<p class='section-label'>Informasi Proyek</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.15);
                border-radius:12px;padding:20px 24px;'>
        <p style='font-size:13px;line-height:1.8;margin:0;color:var(--text-color);'>
            <strong>Nama Proyek:</strong> Bank Marketing Intelligence Dashboard<br>
            <strong>Mata Kuliah:</strong>Data Mining<br>
            <strong>Dosen Pengampu:</strong>Dr. Wiyli Yustanti, S.Si., M.Kom.<br>
             Universitas Negeri Surabaya<br>
            <strong>Tahun:</strong> 2026
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("<p class='section-label'>Informasi Model</p>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.15);
            border-radius:12px;padding:20px 24px;'>
                <p style='font-size:13px;line-height:1.8;margin:0;color:var(--text-color);'>
                    <strong>Model Aktif:</strong> {meta['best_model_name']}<br>
                    <strong>ROC-AUC:</strong> {meta['xgb_metrics']['ROC-AUC']}<br>
                    <strong>Accuracy:</strong> {meta['xgb_metrics']['Accuracy']}<br>
                    <strong>Fitur:</strong> {meta['n_features']}<br>
                    <strong>Cluster:</strong> {meta['n_clusters']}
                </p>
        </div>
        """, unsafe_allow_html=True)
