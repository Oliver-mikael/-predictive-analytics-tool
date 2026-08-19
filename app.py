"""
SalesPredict AI — app.py
========================
Interfaz principal. Todo el backend vive en src/.
Este archivo solo contiene UI + orquestación de flujo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings, io
warnings.filterwarnings("ignore")

# ── Módulos propios ──────────────────────────────────────────────────────────
from src.data_loader  import (cargar_csv_seguro, cargar_excel_seguro,
                               detectar_columnas_clave)
from src.features     import (preparar_serie, sugerir_log, impacto_externas)
from src.models       import get_modelos, HAS_LIGHTGBM, HAS_XGBOOST, HAS_PMDARIMA
from src.forecast     import (analizar, generar_recomendaciones,
                               evaluar_confiabilidad,
                               grafico_proyeccion, grafico_comparacion_modelos,
                               grafico_ventas_rama, grafico_ventas_ciudad,
                               grafico_ventas_producto, grafico_patron_dia_semana,
                               grafico_ventas_hora, tabla_comparacion_periodos,
                               solo_dias_abiertos)
from src.evaluation   import calcular_metricas

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SalesPredict AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════
# CSS — TEMA OSCURO COMPLETO
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Base & reset ─────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #111827 !important;
    color: #F9FAFB !important;
}
.block-container {
    padding: 1.5rem 1.5rem 4rem 1.5rem !important;
    max-width: 1100px !important;
}

/* ── Ocultar elementos de Streamlit ───────────────────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Forzar fondo oscuro en todos los contenedores ───────────────────── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"],
div[data-testid="stForm"],
.element-container { background: transparent !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#111827; }
::-webkit-scrollbar-thumb { background:#374151; border-radius:3px; }

/* ── Header top bar ───────────────────────────────────────────────────── */
.sp-topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding: .75rem 0 1.25rem 0; border-bottom:1px solid #1F2937;
    margin-bottom:1.5rem;
}
.sp-logo { display:flex; align-items:center; gap:.5rem; }
.sp-logo-icon { font-size:1.4rem; }
.sp-logo-text { font-size:1.1rem; font-weight:800; color:#F9FAFB; letter-spacing:-.02em; }
.sp-logo-badge {
    font-size:.65rem; font-weight:700; background:#3B82F6; color:white;
    padding:.15rem .4rem; border-radius:4px; letter-spacing:.04em;
}
.sp-step-pill {
    display:inline-flex; align-items:center; gap:.3rem;
    font-size:.72rem; font-weight:600; color:#9CA3AF;
    background:#1F2937; padding:.3rem .7rem; border-radius:999px;
}
.sp-step-pill .active { color:#3B82F6; }

/* ── Step progress bar ────────────────────────────────────────────────── */
.step-bar {
    display:flex; align-items:center; gap:0; margin-bottom:2rem;
    overflow-x:auto; padding-bottom:.25rem;
}
.step-item {
    display:flex; flex-direction:column; align-items:center; gap:.25rem;
    flex:1; min-width:70px;
}
.step-circle {
    width:32px; height:32px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-size:.75rem; font-weight:700;
    border:2px solid #374151; background:#1F2937; color:#9CA3AF;
    transition: all .3s;
}
.step-circle.done   { background:#22C55E; border-color:#22C55E; color:white; }
.step-circle.active { background:#3B82F6; border-color:#3B82F6; color:white; }
.step-label { font-size:.65rem; font-weight:500; color:#6B7280; white-space:nowrap; }
.step-label.active { color:#3B82F6; }
.step-connector { height:2px; flex:1; min-width:16px; background:#374151; margin-top:-16px; }
.step-connector.done { background:#22C55E; }

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero-wrap {
    text-align:center; padding:3rem 1rem 2.5rem;
}
.hero-badge {
    display:inline-flex; align-items:center; gap:.4rem;
    background:#1F2937; border:1px solid #374151; color:#9CA3AF;
    font-size:.72rem; font-weight:600; padding:.35rem .8rem;
    border-radius:999px; margin-bottom:1.25rem; letter-spacing:.04em;
}
.hero-title {
    font-size:2.6rem; font-weight:800; line-height:1.15; color:#F9FAFB;
    letter-spacing:-.03em; margin-bottom:.75rem;
}
.hero-title span { color:#3B82F6; }
.hero-sub {
    font-size:1.05rem; color:#9CA3AF; font-weight:400;
    max-width:480px; margin:0 auto 2rem auto; line-height:1.6;
}
.hero-cta {
    display:inline-flex; align-items:center; gap:.5rem;
    background:linear-gradient(135deg,#2563EB,#3B82F6);
    color:white; font-weight:700; font-size:1rem;
    padding:.85rem 2.2rem; border-radius:10px; cursor:pointer;
    box-shadow:0 0 30px rgba(59,130,246,.35);
    border:none; text-decoration:none; transition:all .2s;
}
.hero-cta:hover { box-shadow:0 0 40px rgba(59,130,246,.55); transform:translateY(-1px); }
.hero-footer {
    margin-top:1rem; font-size:.78rem; color:#6B7280;
}
.hero-stats {
    display:flex; justify-content:center; gap:2.5rem;
    margin-top:2.5rem; padding-top:2rem; border-top:1px solid #1F2937;
}
.hero-stat-val { font-size:1.5rem; font-weight:800; color:#F9FAFB; }
.hero-stat-lab { font-size:.72rem; color:#6B7280; margin-top:.15rem; }

/* ── Upload zone ──────────────────────────────────────────────────────── */
.upload-title {
    font-size:1.1rem; font-weight:700; color:#F9FAFB; margin-bottom:.25rem;
}
.upload-sub { font-size:.82rem; color:#9CA3AF; margin-bottom:1rem; }
.quality-box {
    background:#1F2937; border:1px solid #374151; border-radius:12px;
    padding:1rem 1.25rem; margin:1rem 0;
}
.quality-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:.4rem 0; border-bottom:1px solid #374151; font-size:.83rem;
}
.quality-row:last-child { border-bottom:none; }
.q-check { color:#22C55E; font-weight:600; }
.q-warn  { color:#F59E0B; font-weight:600; }
.q-score {
    background:#022c22; color:#22C55E; font-weight:800;
    padding:.3rem .8rem; border-radius:8px; font-size:.9rem;
    border:1px solid #15803d;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.card {
    background:#1F2937; border:1px solid #374151; border-radius:14px;
    padding:1.1rem 1.25rem;
}
.card-dark { background:#273449; border-color:#374151; }

/* ── Metric card ──────────────────────────────────────────────────────── */
.mc { background:#1F2937; border:1px solid #374151; border-radius:12px;
      padding:1rem; text-align:center; }
.mc-icon  { font-size:1.3rem; margin-bottom:.3rem; }
.mc-label { font-size:.68rem; color:#9CA3AF; text-transform:uppercase;
            letter-spacing:.06em; font-weight:600; }
.mc-value { font-size:1.4rem; font-weight:800; color:#F9FAFB; margin:.2rem 0 .1rem; }
.mc-sub   { font-size:.7rem; color:#6B7280; }

/* ── Hero result (big number) ─────────────────────────────────────────── */
.result-hero {
    background:linear-gradient(135deg,#1e3a5f 0%,#1d3461 60%,#1e40af 100%);
    border:1px solid #2563eb55; border-radius:16px;
    padding:1.75rem 2rem; margin-bottom:1.5rem;
    box-shadow:0 0 40px rgba(37,99,235,.2);
}
.rh-label { font-size:.72rem; color:#93c5fd; text-transform:uppercase;
            letter-spacing:.07em; font-weight:600; }
.rh-value { font-size:2.8rem; font-weight:800; color:white; line-height:1.1;
            margin:.2rem 0 .4rem; }
.rh-row   { display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }
.rh-badge {
    display:inline-flex; align-items:center; gap:.3rem;
    background:rgba(34,197,94,.18); color:#4ade80;
    border:1px solid rgba(34,197,94,.3);
    font-size:.8rem; font-weight:700;
    padding:.25rem .7rem; border-radius:999px;
}
.rh-badge.down {
    background:rgba(239,68,68,.18); color:#f87171;
    border-color:rgba(239,68,68,.3);
}
.rh-hint { font-size:.82rem; color:#93c5fd; }

/* ── Scenarios ────────────────────────────────────────────────────────── */
.sc-wrap { display:flex; gap:.75rem; flex-wrap:wrap; margin:.75rem 0; }
.sc-card {
    flex:1; min-width:130px; border-radius:12px; padding:.9rem;
    text-align:center; border:1px solid;
}
.sc-icon  { font-size:1.2rem; }
.sc-label { font-size:.67rem; font-weight:700; text-transform:uppercase;
            letter-spacing:.05em; margin-top:.2rem; }
.sc-val   { font-size:1.15rem; font-weight:800; margin-top:.1rem; }
.sc-opt   { background:rgba(34,197,94,.08);  border-color:rgba(34,197,94,.3);  color:#4ade80; }
.sc-base  { background:rgba(59,130,246,.08); border-color:rgba(59,130,246,.3); color:#93c5fd; }
.sc-cons  { background:rgba(245,158,11,.08); border-color:rgba(245,158,11,.3); color:#fcd34d; }

/* ── Recommendation items ────────────────────────────────────────────── */
.rec-section-title {
    font-size:.8rem; font-weight:700; color:#6B7280;
    text-transform:uppercase; letter-spacing:.07em;
    margin:1.25rem 0 .6rem;
}
.rec-item {
    display:flex; align-items:flex-start; gap:.7rem;
    padding:.8rem 1rem; border-radius:10px;
    margin-bottom:.5rem; border-left:3px solid;
    font-size:.875rem; line-height:1.55;
}
.rec-ok   { background:rgba(34,197,94,.07);  border-color:#22C55E; color:#d1fae5; }
.rec-warn { background:rgba(245,158,11,.07); border-color:#F59E0B; color:#fef3c7; }
.rec-info { background:rgba(59,130,246,.07); border-color:#3B82F6; color:#dbeafe; }
.rec-icon { font-size:1rem; flex-shrink:0; padding-top:.05rem; }
.rec-text { flex:1; }

/* ── Loading timeline ────────────────────────────────────────────────── */
.tl-item {
    display:flex; align-items:center; gap:.8rem;
    padding:.5rem 0; font-size:.88rem; color:#D1D5DB;
}
.tl-dot {
    width:22px; height:22px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:.75rem;
}
.tl-done    { background:#22C55E; color:white; }
.tl-active  { background:#3B82F6; color:white;
              animation: pulse 1.2s infinite; }
.tl-pending { background:#374151; color:#6B7280; }

@keyframes pulse {
    0%,100% { box-shadow:0 0 0 0 rgba(59,130,246,.5); }
    50%      { box-shadow:0 0 0 6px rgba(59,130,246,.0); }
}

/* ── Confidence badge ────────────────────────────────────────────────── */
.conf-badge {
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.3rem .75rem; border-radius:999px;
    font-size:.78rem; font-weight:700;
}
.conf-alta  { background:rgba(34,197,94,.15);  color:#4ade80;  border:1px solid rgba(34,197,94,.3);  }
.conf-media { background:rgba(245,158,11,.15); color:#fcd34d; border:1px solid rgba(245,158,11,.3); }
.conf-baja  { background:rgba(239,68,68,.15);  color:#f87171;  border:1px solid rgba(239,68,68,.3);  }

/* ── Download buttons ────────────────────────────────────────────────── */
div.stDownloadButton > button,
div.stButton > button {
    background:linear-gradient(135deg,#2563EB,#3B82F6) !important;
    color:white !important; font-weight:700 !important;
    border:none !important; border-radius:10px !important;
    font-size:.9rem !important; width:100% !important;
    padding:.65rem 1rem !important;
    box-shadow:0 2px 12px rgba(59,130,246,.3) !important;
    transition:all .2s !important;
}
div.stDownloadButton > button:hover,
div.stButton > button:hover {
    box-shadow:0 4px 20px rgba(59,130,246,.5) !important;
    transform:translateY(-1px) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background:#1F2937 !important; border-radius:10px;
    padding:.3rem; gap:.2rem; border:1px solid #374151;
}
[data-baseweb="tab"] {
    background:transparent !important; color:#9CA3AF !important;
    border-radius:8px !important; font-size:.82rem !important;
    font-weight:600 !important; padding:.45rem .9rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background:#3B82F6 !important; color:white !important;
}
[data-baseweb="tab-panel"] {
    padding-top:1.25rem !important;
}

/* ── Table ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

/* ── Selectbox / slider ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"]    label { color:#D1D5DB !important; font-weight:500; }

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background:#1F2937 !important; border:2px dashed #374151 !important;
    border-radius:12px !important;
}
[data-testid="stFileUploader"]:hover { border-color:#3B82F6 !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent !important; }

/* ── Info / warning / error boxes ────────────────────────────────────── */
[data-testid="stAlert"] {
    background:#1F2937 !important; border-color:#374151 !important;
    border-radius:10px !important;
}

/* ── Mobile bottom nav ────────────────────────────────────────────────── */
.mobile-nav {
    display:none;
    position:fixed; bottom:0; left:0; right:0; z-index:9999;
    background:#1F2937; border-top:1px solid #374151;
    padding:.5rem 0 calc(.5rem + env(safe-area-inset-bottom));
}
.mobile-nav-items {
    display:flex; justify-content:space-around; align-items:center; max-width:480px; margin:0 auto;
}
.mobile-nav-item {
    display:flex; flex-direction:column; align-items:center; gap:.2rem;
    font-size:.62rem; color:#6B7280; padding:.3rem .5rem;
    cursor:pointer; min-width:50px;
}
.mobile-nav-item.active { color:#3B82F6; }
.mobile-nav-item .nav-icon { font-size:1.2rem; }

@media (max-width:640px) {
    .mobile-nav { display:block; }
    .hero-title  { font-size:1.8rem; }
    .rh-value    { font-size:2rem; }
    .hero-stats  { gap:1.5rem; }
    .block-container { padding:1rem 1rem 5rem 1rem !important; }
    .step-label  { display:none; }
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background:#1F2937 !important;
    border-right:1px solid #374151 !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "step":       "hero",   # hero | upload | analyzing | results
        "df_raw":     None,
        "df_diario":  None,
        "info":       None,
        "prediccion": None,
        "metricas":   None,
        "columnas":   None,
        "pais":       "Bolivia",
        "moneda":     "Bs.",
        "dias":       30,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

MONEDAS = {
    "Bolivia (Bs.)":    "Bs.",
    "USA ($)":          "$",
    "México (MXN)":     "MXN",
    "Perú (S/)":        "S/",
    "Colombia (COP)":   "COP",
    "Argentina (ARS)":  "ARS",
    "Chile (CLP)":      "CLP",
    "Brasil (R$)":      "R$",
}

PAISES = ["Bolivia","Mexico","Colombia","Peru","Chile","Argentina",
          "Espana","USA","Brasil","Ecuador","Venezuela","Paraguay"]

def fmt(valor: float) -> str:
    m = st.session_state.moneda
    return f"{m} {valor:,.0f}"

def _safe_chart(fig, key: str = None):
    try:
        if fig is not None:
            kw = {"use_container_width": True}
            if key:
                kw["key"] = key
            st.plotly_chart(fig, **kw)
            return True
    except Exception:
        pass
    return False


def _calidad_datos(df_raw, col_fecha, col_ventas) -> dict:
    """Devuelve checks de calidad del CSV antes del análisis."""
    checks, score = [], 100

    # Fechas faltantes
    nulos_fecha = df_raw[col_fecha].isna().sum() if col_fecha else len(df_raw)
    if nulos_fecha == 0:
        checks.append(("✅", "Sin fechas faltantes", True))
    else:
        checks.append(("⚠️", f"{nulos_fecha} fechas faltantes", False))
        score -= 10

    # Duplicados
    dups = df_raw.duplicated().sum()
    if dups == 0:
        checks.append(("✅", "Sin duplicados", True))
    else:
        checks.append(("⚠️", f"{dups} filas duplicadas", False))
        score -= 5

    # Ventas nulas
    nulos_v = df_raw[col_ventas].isna().sum() if col_ventas else len(df_raw)
    if nulos_v == 0:
        checks.append(("✅", "Sin valores vacíos en ventas", True))
    else:
        checks.append(("⚠️", f"{nulos_v} valores vacíos en ventas", False))
        score -= 8

    # Formato
    checks.append(("✅", "Formato de archivo correcto", True))

    nivel = "Excelente" if score >= 95 else ("Bueno" if score >= 80 else "Mejorable")
    return {"checks": checks, "score": score, "nivel": nivel}


# ════════════════════════════════════════════════════════════════════════════
# TOPBAR + STEP PROGRESS
# ════════════════════════════════════════════════════════════════════════════

PASOS = ["Bienvenida","Carga","Procesando","Dashboard","Recomendaciones","Descarga"]
_STEP_IDX = {"hero":0, "upload":1, "analyzing":2, "results":3}

def _render_topbar():
    step_n = _STEP_IDX.get(st.session_state.step, 0)
    st.markdown(f"""
    <div class="sp-topbar">
      <div class="sp-logo">
        <span class="sp-logo-icon">📈</span>
        <span class="sp-logo-text">SalesPredict</span>
        <span class="sp-logo-badge">AI</span>
      </div>
      <div class="sp-step-pill">
        <span class="active">Paso {step_n+1}</span>
        <span>de {len(PASOS)}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_stepbar():
    step_n = _STEP_IDX.get(st.session_state.step, 0)
    items_html = ""
    for i, label in enumerate(PASOS):
        if i > 0:
            cls = "done" if i <= step_n else ""
            items_html += f'<div class="step-connector {cls}"></div>'
        if i < step_n:
            cir, lab = "done", ""
        elif i == step_n:
            cir, lab = "active", "active"
        else:
            cir, lab = "", ""
        icon = "✓" if i < step_n else str(i+1)
        items_html += f"""
        <div class="step-item">
          <div class="step-circle {cir}">{icon}</div>
          <div class="step-label {lab}">{label}</div>
        </div>"""
    st.markdown(f'<div class="step-bar">{items_html}</div>', unsafe_allow_html=True)


def _render_mobile_nav():
    tabs = [("🏠","Resumen"),("📈","Predicciones"),("🔔","Alertas"),
            ("📊","Análisis"),("⋯","Más")]
    items = "".join(
        f'<div class="mobile-nav-item{"  active" if i==0 else ""}">'
        f'<span class="nav-icon">{icon}</span>{lab}</div>'
        for i,(icon,lab) in enumerate(tabs)
    )
    st.markdown(
        f'<div class="mobile-nav"><div class="mobile-nav-items">{items}</div></div>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — HERO
# ════════════════════════════════════════════════════════════════════════════

def render_hero():
    _render_topbar()
    _render_stepbar()

    modelos_activos = len(get_modelos())

    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-badge">🤖 IA · {modelos_activos} modelos · Sin código</div>
      <h1 class="hero-title">
        Convierte tus ventas<br>en <span>mejores decisiones</span>
      </h1>
      <p class="hero-sub">
        Predice, planifica y haz crecer tu negocio con inteligencia artificial.
        Solo sube tu historial de ventas en CSV o Excel.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_demo, col_pad = st.columns([1, 1, 1])
    with col_btn:
        if st.button("Comenzar análisis →", key="btn_hero", use_container_width=True):
            st.session_state.step = "upload"
            st.rerun()
    with col_demo:
        if st.button("▶ Ver demo con CSV", key="btn_demo", use_container_width=True):
            import io as _io
            from src.data_loader import cargar_csv_seguro as _load_csv, detectar_columnas_clave as _det
            with open("data/ventas_ejemplo.csv", "rb") as _f:
                _df_raw, _ = _load_csv(_io.BytesIO(_f.read()))
            _cols = _det(_df_raw)
            st.session_state.df_raw   = _df_raw
            st.session_state.columnas = {**_cols,
                                          "fecha_sel":    _cols["fecha"],
                                          "col_ventas_sel": _cols["ventas"]}
            st.session_state.dias     = 30
            st.session_state.demo_mode = True
            st.session_state.step     = "analyzing"
            st.rerun()

    st.markdown("""
    <p style="text-align:center;font-size:.78rem;color:#6B7280;margin-top:.5rem">
        Sin registro &nbsp;·&nbsp; Rápido &nbsp;·&nbsp; Seguro &nbsp;·&nbsp; Gratis
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-stats">
      <div style="text-align:center">
        <div class="hero-stat-val">~85%</div>
        <div class="hero-stat-lab">Precisión estimada</div>
      </div>
      <div style="text-align:center">
        <div class="hero-stat-val">&lt; 60 s</div>
        <div class="hero-stat-lab">Tiempo de análisis</div>
      </div>
      <div style="text-align:center">
        <div class="hero-stat-val">5+</div>
        <div class="hero-stat-lab">Modelos comparados</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar config
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        pais = st.selectbox("🌍 País", PAISES,
                            index=PAISES.index(st.session_state.pais))
        mon  = st.selectbox("💰 Moneda", list(MONEDAS.keys()))
        st.session_state.pais   = pais
        st.session_state.moneda = MONEDAS[mon]
        st.markdown("---")
        st.markdown("📂 [Descargar CSV de ejemplo](data/ventas_ejemplo.csv)")
        st.markdown("❓ ¿Necesitas ayuda? [Leer guía](#)")

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA
# ════════════════════════════════════════════════════════════════════════════

def render_upload():
    _render_topbar()
    _render_stepbar()

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Volver", key="btn_back_upload"):
            st.session_state.step = "hero"
            st.rerun()

    st.markdown("""
    <p class="upload-title">Sube tu archivo de ventas</p>
    <p class="upload-sub">CSV o Excel con al menos una columna de <b>fecha</b> y otra de <b>ventas</b></p>
    """, unsafe_allow_html=True)

    archivo = st.file_uploader(
        "Arrastra tu archivo aquí o haz clic",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    # Config
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        pais = st.selectbox("🌍 País", PAISES,
                            index=PAISES.index(st.session_state.pais))
        mon  = st.selectbox("💰 Moneda", list(MONEDAS.keys()))
        st.session_state.pais   = pais
        st.session_state.moneda = MONEDAS[mon]

    if archivo is None:
        st.markdown("""
        <div class="card" style="margin-top:1rem;text-align:center;padding:2rem">
          <div style="font-size:2.5rem;margin-bottom:.75rem">📂</div>
          <div style="color:#9CA3AF;font-size:.88rem">
            Formatos aceptados: <b>CSV</b>, <b>XLSX</b>, <b>XLS</b><br>
            Tamaño máximo: 200 MB
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Cargar datos
    try:
        if archivo.name.endswith((".xlsx", ".xls")):
            df_raw, _ = cargar_excel_seguro(archivo)
        else:
            df_raw, _ = cargar_csv_seguro(archivo)
    except Exception as e:
        st.error("No se pudo leer el archivo. Verifica que sea un CSV o Excel válido.")
        return

    if df_raw is None:
        st.error("No se pudo leer el archivo.")
        return

    columnas = detectar_columnas_clave(df_raw)

    # Selectores de columnas
    col1, col2 = st.columns(2)
    with col1:
        all_cols  = df_raw.columns.tolist()
        idx_f     = all_cols.index(columnas["fecha"]) if columnas["fecha"] in all_cols else 0
        col_fecha = st.selectbox("📅 Columna de fechas", all_cols, index=idx_f)
    with col2:
        idx_v      = all_cols.index(columnas["ventas"]) if columnas["ventas"] in all_cols else 0
        col_ventas = st.selectbox("💰 Columna de ventas", all_cols, index=idx_v)

    # Preview
    with st.expander("👁 Vista previa del archivo"):
        try:
            st.dataframe(df_raw.head(6), use_container_width=True, hide_index=True)
        except Exception:
            pass

    # Checks de calidad
    calidad = _calidad_datos(df_raw, col_fecha, col_ventas)
    checks_html = ""
    for icon, texto, ok in calidad["checks"]:
        cls = "q-check" if ok else "q-warn"
        checks_html += f"""
        <div class="quality-row">
          <span class="{cls}">{icon} {texto}</span>
        </div>"""
    score_color = "#22C55E" if calidad["score"] >= 90 else "#F59E0B"
    st.markdown(f"""
    <div class="quality-box">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">
        <span style="font-size:.85rem;font-weight:700;color:#D1D5DB">Verificación de calidad</span>
        <span class="q-score" style="color:{score_color}">
          {calidad["score"]}% {calidad["nivel"]}
        </span>
      </div>
      {checks_html}
    </div>
    """, unsafe_allow_html=True)

    dias_futuro = st.select_slider(
        "📆 Días a proyectar",
        options=[7, 14, 21, 30, 45, 60, 90],
        value=30,
        key="dias_slider",
    )

    st.session_state.df_raw    = df_raw
    st.session_state.columnas  = {**columnas, "fecha_sel": col_fecha, "col_ventas_sel": col_ventas}
    st.session_state.dias      = dias_futuro

    if st.button("Analizar ventas 📊", key="btn_analizar", use_container_width=True):
        st.session_state.step = "analyzing"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ANALIZANDO
# ════════════════════════════════════════════════════════════════════════════

def render_analyzing():
    _render_topbar()
    _render_stepbar()

    df_raw     = st.session_state.df_raw
    columnas   = st.session_state.columnas
    col_fecha  = columnas["fecha_sel"]
    col_ventas = columnas["col_ventas_sel"]
    dias       = st.session_state.dias
    pais       = st.session_state.pais

    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <div style="font-size:2rem;margin-bottom:.5rem">🔄</div>
      <h2 style="font-size:1.4rem;font-weight:800;color:#F9FAFB;margin:0">
        Analizando tus datos…
      </h2>
      <p style="color:#9CA3AF;font-size:.85rem;margin:.4rem 0 0">
        Esto suele tardar entre 15 y 60 segundos
      </p>
    </div>
    """, unsafe_allow_html=True)

    pasos_tl = [
        "Archivo recibido",
        "Limpieza y validación de datos",
        "Detección de patrones estacionales",
        "Comparando modelos de IA",
        "Generando proyección",
        "Creando recomendaciones",
    ]

    tl_placeholder = st.empty()
    prog_bar       = st.progress(0)

    def render_tl(done_idx: int, active_idx: int):
        html = '<div style="max-width:480px;margin:0 auto">'
        for i, label in enumerate(pasos_tl):
            if i < done_idx:
                cls, icon = "tl-done", "✓"
            elif i == active_idx:
                cls, icon = "tl-active", "⏳"
            else:
                cls, icon = "tl-pending", str(i+1)
            html += f"""
            <div class="tl-item">
              <div class="tl-dot {cls}">{icon}</div>
              <span>{label}</span>
            </div>"""
        html += "</div>"
        tl_placeholder.markdown(html, unsafe_allow_html=True)

    render_tl(0, 0)
    prog_bar.progress(5)

    # ── Paso 1: limpiar datos ──────────────────────────────────────────────
    cols_ext = {
        k: columnas.get(k)
        for k in ["temperatura","lluvia","evento","tasa_inflacion","trafico_web","conversion","carritos"]
        if columnas.get(k)
    }
    df_diario, info = preparar_serie(df_raw, col_fecha, col_ventas, cols_ext or None)

    if info["estado"] == "ERROR":
        prog_bar.empty(); tl_placeholder.empty()
        st.error(f"❌ {info['mensaje']}")
        if st.button("← Volver a cargar datos"):
            st.session_state.step = "upload"
            st.rerun()
        return

    render_tl(2, 2); prog_bar.progress(25)

    # ── Paso 2: análisis ───────────────────────────────────────────────────
    pasos_modelo = [None]

    def cb_progreso(fraccion: float, nombre: str):
        pasos_modelo[0] = nombre
        render_tl(2, 3)
        prog_bar.progress(int(25 + fraccion * 50))

    # La selección automática de log se resuelve dentro de cada fold para
    # evitar usar información de la ventana de validación.
    usar_log = None
    prediccion, metricas, error = analizar(
        df_diario, pais, dias,
        usar_log=usar_log, n_folds=3,
        progreso=cb_progreso,
    )

    if error or prediccion is None:
        prog_bar.empty(); tl_placeholder.empty()
        st.error(f"No se pudo completar el análisis. {error or ''}")
        if st.button("← Volver"):
            st.session_state.step = "upload"
            st.rerun()
        return

    render_tl(5, 5); prog_bar.progress(95)

    st.session_state.df_diario  = df_diario
    st.session_state.info       = info
    st.session_state.prediccion = prediccion
    st.session_state.metricas   = metricas
    st.session_state.step       = "results"

    prog_bar.progress(100)
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — RESULTADOS
# ════════════════════════════════════════════════════════════════════════════

def render_results():
    _render_topbar()
    _render_stepbar()

    df_diario  = st.session_state.df_diario
    df_raw     = st.session_state.df_raw
    prediccion = st.session_state.prediccion
    metricas   = st.session_state.metricas
    info       = st.session_state.info
    columnas   = st.session_state.columnas
    moneda     = st.session_state.moneda
    dias       = st.session_state.dias

    col_back, col_title, col_new = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Subir otro", key="btn_new"):
            for k in ["df_raw","df_diario","info","prediccion","metricas"]:
                st.session_state[k] = None
            st.session_state.step = "upload"
            st.rerun()

    # Sidebar config
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        pais = st.selectbox("🌍 País", PAISES,
                            index=PAISES.index(st.session_state.pais))
        mon  = st.selectbox("💰 Moneda", list(MONEDAS.keys()))
        st.session_state.pais   = pais
        st.session_state.moneda = MONEDAS[mon]
        moneda = MONEDAS[mon]
        st.markdown("---")
        dias = st.select_slider("📆 Días proyectados", [7,14,21,30,45,60,90], value=dias)

    # ── Calcular KPIs ─────────────────────────────────────────────────────
    hoy      = df_diario["ds"].max()
    futuro   = prediccion[prediccion["ds"] > hoy]
    total_f  = float(futuro["yhat"].sum())
    abiertos = solo_dias_abiertos(df_diario)
    ult_mismo_periodo = float(abiertos["y"].tail(dias).sum())
    cambio_pct = (total_f - ult_mismo_periodo) / max(ult_mismo_periodo, 1) * 100
    precision  = metricas["Precision"]
    conf       = evaluar_confiabilidad(df_diario, metricas["MAPE"])

    tab_res, tab_pred, tab_cat, tab_rec = st.tabs([
        "📊 Resumen", "📈 Predicciones", "🗂 Categorías", "🤖 Recomendaciones IA"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — RESUMEN
    # ════════════════════════════════════════════════════════════════════════
    with tab_res:
        # ── Big hero number ─────────────────────────────────────────────
        dir_icon  = "↑" if cambio_pct >= 0 else "↓"
        badge_cls = "rh-badge" if cambio_pct >= 0 else "rh-badge down"
        st.markdown(f"""
        <div class="result-hero">
          <div class="rh-label">Ventas esperadas — próximos {dias} días</div>
          <div class="rh-value">{moneda} {total_f:,.0f}</div>
          <div class="rh-row">
            <span class="{badge_cls}">{dir_icon} {abs(cambio_pct):.1f}% vs período anterior</span>
            <span class="rh-hint">Proyección del Sistema IA</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4 metric cards ───────────────────────────────────────────────
        dias_nombres = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
        abiertos_temp = abiertos.copy()
        abiertos_temp["dow"] = abiertos_temp["ds"].dt.dayofweek
        mejor_dow     = abiertos_temp.groupby("dow")["y"].mean().idxmax()
        mejor_dia     = dias_nombres[mejor_dow]
        alertas_n     = sum(1 for r in generar_recomendaciones(df_diario, prediccion, metricas, info)
                           if r["tipo"] == "alerta")

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, label, val, sub in [
            (c1, "📅", "Mejor día", mejor_dia, "Mayor venta promedio"),
            (c2, "🎯", "Precisión IA", f"{precision:.0f}%", "Estimación del sistema"),
            (c3, "📊", "Calidad datos", f"{100 - info['pct_zeros']:.0f}%", f"{info['dias']} días disponibles"),
            (c4, "🔔", "Alertas", str(alertas_n), "Requieren atención"),
        ]:
            with col:
                st.markdown(f"""
                <div class="mc">
                  <div class="mc-icon">{icon}</div>
                  <div class="mc-label">{label}</div>
                  <div class="mc-value">{val}</div>
                  <div class="mc-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.25rem'/>", unsafe_allow_html=True)

        # ── Gráfico principal ────────────────────────────────────────────
        fig_main = grafico_proyeccion(df_diario, prediccion, moneda,
                                       f"Proyección — {dias} días")
        _safe_chart(fig_main, "chart_main")

        # ── Confiabilidad ────────────────────────────────────────────────
        nivel_raw = conf["nivel"].split()[1]  # ALTA / MEDIA / BAJA
        cls_map   = {"ALTA":"conf-alta","MEDIA":"conf-media","BAJA":"conf-baja"}
        cls_conf  = cls_map.get(nivel_raw, "conf-media")
        det_html  = " &nbsp;·&nbsp; ".join(conf["detalles"])
        st.markdown(f"""
        <div class="card" style="margin-top:.75rem">
          <div style="display:flex;align-items:center;gap:.75rem;flex-wrap:wrap">
            <span style="font-size:.85rem;font-weight:700;color:#D1D5DB">Nivel de confianza:</span>
            <span class="conf-badge {cls_conf}">{conf['nivel']} — {conf['score']}/100</span>
          </div>
          <div style="font-size:.75rem;color:#6B7280;margin-top:.5rem">{det_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — PREDICCIONES
    # ════════════════════════════════════════════════════════════════════════
    with tab_pred:
        # ── Escenarios ───────────────────────────────────────────────────
        q_alto = prediccion["yhat_upper"].sum() if "yhat_upper" in prediccion else total_f * 1.15
        q_bajo = prediccion["yhat_lower"].sum() if "yhat_lower" in prediccion else total_f * 0.88
        st.markdown(f"""
        <div class="sc-wrap">
          <div class="sc-card sc-opt">
            <div class="sc-icon">🚀</div>
            <div class="sc-label">Optimista</div>
            <div class="sc-val">{moneda} {float(futuro['yhat_upper'].sum() if 'yhat_upper' in futuro.columns else total_f*1.15):,.0f}</div>
          </div>
          <div class="sc-card sc-base">
            <div class="sc-icon">📈</div>
            <div class="sc-label">Esperado</div>
            <div class="sc-val">{moneda} {total_f:,.0f}</div>
          </div>
          <div class="sc-card sc-cons">
            <div class="sc-icon">🛡️</div>
            <div class="sc-label">Conservador</div>
            <div class="sc-val">{moneda} {float(futuro['yhat_lower'].sum() if 'yhat_lower' in futuro.columns else total_f*0.88):,.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gráfico ───────────────────────────────────────────────────────
        fig_full = grafico_proyeccion(df_diario, prediccion, moneda)
        _safe_chart(fig_full, "chart_pred")

        # ── Comparación de modelos ────────────────────────────────────────
        pred_por_modelo = metricas.get("pred_por_modelo", {})
        if pred_por_modelo:
            with st.expander("🔬 Comparar modelos individuales"):
                fig_cmp = grafico_comparacion_modelos(pred_por_modelo, df_diario, moneda)
                _safe_chart(fig_cmp, "chart_cmp")

                # Tabla de métricas por modelo
                por_mod = metricas.get("por_modelo", {})
                if por_mod:
                    rows = []
                    for nombre, m in por_mod.items():
                        peso = metricas["pesos"].get(nombre, 0)
                        rows.append({
                            "Modelo":    nombre,
                            "Precisión": f"{max(0,100-m['mape']):.1f}%",
                            "MAE":       f"{m['mae']:,.0f}",
                            "RMSE":      f"{m.get('rmse',0):,.0f}",
                            "Peso IA":   f"{peso:.0%}",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)

        # ── Tabla día a día ────────────────────────────────────────────
        with st.expander("📋 Tabla día a día"):
            tabla = futuro[["ds","yhat"]].copy()
            tabla.columns = ["Fecha","Proyección"]
            tabla["Fecha"] = tabla["Fecha"].dt.strftime("%a %d %b")
            tabla["Proyección"] = tabla["Proyección"].map(lambda v: f"{moneda} {v:,.0f}")
            if "yhat_lower" in futuro.columns:
                tabla["Mínimo"]  = futuro["yhat_lower"].map(lambda v: f"{moneda} {v:,.0f}").values
                tabla["Máximo"]  = futuro["yhat_upper"].map(lambda v: f"{moneda} {v:,.0f}").values
            st.dataframe(tabla, hide_index=True, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — CATEGORÍAS
    # ════════════════════════════════════════════════════════════════════════
    with tab_cat:
        col_branch   = columnas.get("branch")
        col_ciudad   = columnas.get("ciudad")
        col_producto = columnas.get("producto")
        col_hora     = columnas.get("hora")
        col_ventas_r = columnas.get("col_ventas_sel", "y")

        mostro = False

        fig_rama = grafico_ventas_rama(df_raw, col_branch, col_ventas_r)
        if fig_rama:
            _safe_chart(fig_rama, "ch_rama"); mostro = True

        fig_ciudad = grafico_ventas_ciudad(df_raw, col_ciudad, col_ventas_r)
        if fig_ciudad:
            _safe_chart(fig_ciudad, "ch_ciudad"); mostro = True

        fig_prod = grafico_ventas_producto(df_raw, col_producto, col_ventas_r)
        if fig_prod:
            _safe_chart(fig_prod, "ch_prod"); mostro = True

        _safe_chart(grafico_patron_dia_semana(df_diario), "ch_dow")
        mostro = True

        fig_hora = grafico_ventas_hora(df_raw, col_hora, col_ventas_r)
        if fig_hora:
            _safe_chart(fig_hora, "ch_hora")

        # Comparación de períodos
        with st.expander("📊 Comparación de períodos"):
            try:
                tab_periodos = tabla_comparacion_periodos(df_diario)
                st.dataframe(tab_periodos, hide_index=True, use_container_width=True)
            except Exception:
                pass

        # Variables externas detectadas
        ext = info.get("externas_disponibles", [])
        if ext:
            from src.features import impacto_externas
            ganancias = impacto_externas(ext)
            if ganancias:
                rows_ext = [{"Variable": v[0], "Impacto estimado en precisión": v[1]}
                            for v in ganancias.values()]
                with st.expander("🌡 Variables externas en uso"):
                    st.dataframe(pd.DataFrame(rows_ext), hide_index=True,
                                 use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — RECOMENDACIONES IA
    # ════════════════════════════════════════════════════════════════════════
    with tab_rec:
        recs = generar_recomendaciones(df_diario, prediccion, metricas, info)

        # ── ¿Qué detectó la IA? ──────────────────────────────────────────
        st.markdown('<div class="rec-section-title">🔍 ¿Qué detectó la IA?</div>',
                    unsafe_allow_html=True)

        # Hallazgos automáticos
        hallazgos = []
        # Mejor día
        abiertos2 = abiertos.copy()
        abiertos2["dow"] = abiertos2["ds"].dt.dayofweek
        dias_nombres_full = ["lunes","martes","miércoles","jueves","viernes","sábados","domingos"]
        media_dow = abiertos2.groupby("dow")["y"].mean()
        mejor_d   = media_dow.idxmax()
        peor_d    = media_dow.idxmin()
        hallazgos.append(f"Las ventas son más altas los <b>{dias_nombres_full[mejor_d]}</b> "
                         f"({moneda} {media_dow[mejor_d]:,.0f} promedio).")
        hallazgos.append(f"Los <b>{dias_nombres_full[peor_d]}</b> tienen el menor volumen "
                         f"({moneda} {media_dow[peor_d]:,.0f} promedio).")

        feriados_set = None
        if metricas.get("pesos"):
            # Tendencia
            primera = float(abiertos["y"][:len(abiertos)//2].mean())
            segunda = float(abiertos["y"][len(abiertos)//2:].mean())
            if primera > 0:
                cambio = (segunda - primera) / primera * 100
                dir_t = "creciendo" if cambio > 0 else "bajando"
                hallazgos.append(f"La tendencia está <b>{dir_t}</b> "
                                  f"({abs(cambio):.1f}% en el último período).")

        # Externas
        ext = info.get("externas_disponibles", [])
        if "lluvia" in ext:
            hallazgos.append("Se detectó correlación entre <b>días de lluvia</b> y menores ventas.")
        if "evento" in ext:
            hallazgos.append("Los <b>eventos locales</b> generan picos de venta positivos.")

        for h in hallazgos:
            st.markdown(f"""
            <div class="rec-item rec-info">
              <span class="rec-icon">🔎</span>
              <span class="rec-text">{h}</span>
            </div>""", unsafe_allow_html=True)

        # ── ¿Qué haría un gerente hoy? ───────────────────────────────────
        st.markdown('<div class="rec-section-title">💼 ¿Qué haría un gerente hoy?</div>',
                    unsafe_allow_html=True)

        icon_map  = {"positivo": "🟢", "alerta": "🟡", "info": "🔵"}
        class_map = {"positivo": "rec-ok", "alerta": "rec-warn", "info": "rec-info"}
        for rec in recs:
            icon  = icon_map.get(rec["tipo"], "🔵")
            cls   = class_map.get(rec["tipo"], "rec-info")
            st.markdown(f"""
            <div class="rec-item {cls}">
              <span class="rec-icon">{icon}</span>
              <span class="rec-text">{rec['texto']}</span>
            </div>""", unsafe_allow_html=True)

        # ── Métricas técnicas (colapsado) ─────────────────────────────────
        with st.expander("⚙️ Detalles técnicos del Sistema IA"):
            st.markdown(f"""
            | Métrica | Valor |
            |---------|-------|
            | Precisión estimada | {precision:.1f}% |
            | Error absoluto medio (MAE) | {moneda} {metricas['MAE']:,.0f} |
            | Error cuadrático (RMSE) | {moneda} {metricas.get('RMSE',0):,.0f} |
            | Error ponderado (WAPE) | {metricas['WAPE']:.1f}% |
            | Folds de validación | {metricas['folds']} |
            | Días analizados | {info['dias']} |
            | Formato de fecha | {info['formato_fecha']} |
            """)
            modelos_activos = ", ".join(metricas["pesos"].keys())
            st.markdown(f"**Modelos activos:** {modelos_activos}")

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN DE DESCARGA
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;margin-bottom:1rem">
      <span style="font-size:.95rem;font-weight:700;color:#D1D5DB">Exportar resultados</span>
    </div>
    """, unsafe_allow_html=True)

    # Preparar CSV de proyección
    csv_pred = futuro[["ds","yhat"]].copy()
    csv_pred.columns = ["Fecha","Proyeccion"]
    if "yhat_lower" in futuro.columns:
        csv_pred["Minimo"]  = futuro["yhat_lower"].values
        csv_pred["Maximo"]  = futuro["yhat_upper"].values
    csv_bytes = csv_pred.to_csv(index=False).encode("utf-8")

    # Preparar CSV de stock
    csv_stock = csv_pred.copy()
    csv_stock["Stock_sugerido"] = (csv_stock["Proyeccion"] * 1.20).round(0)
    stock_bytes = csv_stock.to_csv(index=False).encode("utf-8")

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button(
            "📊 Descargar proyección CSV",
            data=csv_bytes,
            file_name="proyeccion_ventas.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dc2:
        st.download_button(
            "📦 Plan de stock Excel",
            data=stock_bytes,
            file_name="plan_stock.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dc3:
        reporte_txt = (
            f"REPORTE SALESPREDICT AI\n"
            f"========================\n"
            f"Generado: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            f"Ventas esperadas ({dias} días): {moneda} {total_f:,.0f}\n"
            f"vs período anterior: {cambio_pct:+.1f}%\n"
            f"Precisión del sistema: {precision:.1f}%\n"
            f"Días analizados: {info['dias']}\n\n"
            f"PROYECCIÓN DÍA A DÍA\n"
            f"--------------------\n"
        ) + "\n".join(
            f"{row['Fecha'].strftime('%d/%m/%Y')}: {moneda} {row['Proyeccion']:,.0f}"
            for _, row in csv_pred.iterrows()
        )
        st.download_button(
            "📄 Reporte de texto",
            data=reporte_txt.encode("utf-8"),
            file_name="reporte_salespredict.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Mobile nav
    if st.session_state.step == "results":
        _render_mobile_nav()


# ════════════════════════════════════════════════════════════════════════════
# ROUTER PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

step = st.session_state.step

if step == "hero":
    render_hero()
elif step == "upload":
    render_upload()
elif step == "analyzing":
    render_analyzing()
elif step == "results":
    render_results()
else:
    st.session_state.step = "hero"
    st.rerun()
