from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure project `src/` wins module resolution (Streamlit reruns can otherwise
# pick up a different module with the same name earlier on sys.path).
src_str = str(SRC_DIR)
if src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)

# Import as module to avoid Streamlit's occasional stale from-import issues
import inference_engine as ie  # noqa

predict_dataframe = ie.predict_dataframe
available_models = getattr(ie, "available_models", lambda: ["Default"])
DEFAULT_MODEL_NAME = getattr(ie, "DEFAULT_MODEL_NAME", "Default")

FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "random_forest_feature_importance.csv"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — dark cyber aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #080c12;
    color: #c9d8e8;
}

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* ── hero title ── */
.ids-hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.25rem;
}
.ids-hero h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    color: #00e5ff;
    letter-spacing: 0.08em;
    margin: 0;
    text-shadow: 0 0 24px rgba(0,229,255,0.35);
}
.ids-tagline {
    font-size: 1rem;
    color: #5e7a92;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* ── section card ── */
.step-card {
    background: linear-gradient(135deg, #0d1621 0%, #111c2a 100%);
    border: 1px solid #1a2d42;
    border-left: 3px solid #00e5ff;
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.6rem;
    position: relative;
}
.step-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #00e5ff;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.step-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e0eaf4;
    margin-bottom: 1rem;
}

/* ── metric tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.6rem;
}
.metric-tile {
    background: #0d1621;
    border: 1px solid #1a2d42;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00e5ff, #0066ff);
}
.metric-label {
    font-size: 0.78rem;
    color: #5e7a92;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00e5ff;
    line-height: 1;
}
.metric-value.alert { color: #ff4b6e; }
.metric-value.warn  { color: #f5a623; }
.metric-value.ok    { color: #00e5a0; }

/* ── section headings inside results ── */
.section-heading {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #00e5ff;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2d42;
    padding-bottom: 0.5rem;
    margin: 1.8rem 0 1rem;
}

/* ── file uploader tweak ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1a3a52 !important;
    border-radius: 8px !important;
    background: #0a111a !important;
    padding: 0.5rem !important;
}

/* ── buttons ── */
.stButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: linear-gradient(135deg, #003d66, #005c99) !important;
    color: #00e5ff !important;
    border: 1px solid #00749c !important;
    border-radius: 6px !important;
    letter-spacing: 0.1em !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #005c99, #0080cc) !important;
    border-color: #00e5ff !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.25) !important;
}
.stButton > button:disabled {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
}

/* ── download button ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: #0d1f10 !important;
    color: #00e5a0 !important;
    border: 1px solid #00603a !important;
    border-radius: 6px !important;
    letter-spacing: 0.08em !important;
}

/* ── progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #003d66, #00e5ff) !important;
    border-radius: 4px !important;
}

/* ── alerts / success ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 4px !important;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2d42 !important;
    border-radius: 8px !important;
}

/* ── warning notice ── */
.upload-notice {
    background: #1a1200;
    border: 1px solid #665200;
    border-left: 3px solid #f5a623;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.9rem;
    color: #c8a84b;
    margin-top: 0.8rem;
}

/* ── status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.12em;
}
.badge-ready  { background: #0d2a1a; color: #00e5a0; border: 1px solid #00603a; }
.badge-idle   { background: #0d1621; color: #5e7a92; border: 1px solid #1a2d42; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATE INIT
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="ids-hero">
    <h1>🛡️ INTRUSION DETECTION SYSTEM</h1>
</div>
<div class="ids-tagline">Upload network traffic data &nbsp;·&nbsp; Run ML detection &nbsp;·&nbsp; Analyze threat alerts</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP 1 — Upload
# ─────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 01</div>
    <div class="step-title">Upload Dataset</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your CSV file here (up to 400 MB)",
    type=["csv"],
    help="Large files are processed in streaming chunks — no memory issues.",
)

if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    st.success(f"✔ **{uploaded_file.name}** loaded — {size_mb:.1f} MB")
    st.session_state.file_loaded = True
else:
    st.markdown("""
    <div class="upload-notice">
        ⚠ Large files (100 – 400 MB) are supported via chunk-streaming.
        Processing time scales with file size.
    </div>
    """, unsafe_allow_html=True)
    st.session_state.file_loaded = False

# ─────────────────────────────────────────────
# STEP 2 — Run
# ─────────────────────────────────────────────
st.markdown("""
<div class="step-card" style="margin-top:1.6rem;">
    <div class="step-label">Step 02</div>
    <div class="step-title">Run Detection Engine</div>
</div>
""", unsafe_allow_html=True)

model_name = st.selectbox(
    "Choose model",
    options=available_models(),
    index=available_models().index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in available_models() else 0,
    help="Pick the model to run. Hierarchical Family + Subtype is the recommended final model; Improved GPU LightGBM remains available as the strongest flat-model comparison.",
)

run_btn = st.button(
    "⚡  RUN DETECTION",
    disabled=not st.session_state.file_loaded,
)

# ─────────────────────────────────────────────
# ATTACK SIMULATION PANEL — manual single sample
# ─────────────────────────────────────────────
st.markdown("""
<div class="step-card" style="margin-top:1.2rem;">
    <div class="step-label">Attack Simulation Panel</div>
    <div class="step-title">Manual Single‑Sample Prediction</div>
</div>
""", unsafe_allow_html=True)

with st.expander("Open simulation inputs", expanded=False):
    st.markdown(
        "<span style='color:#5e7a92'>Enter feature values for one synthetic flow, then run the selected model.</span>",
        unsafe_allow_html=True,
    )

    # Reasonable defaults: zeros across the board (user can modify)
    sim_values: dict[str, float] = {}
    sim_cols = list(getattr(ie, "EXPECTED_COLUMNS", []))

    if not sim_cols:
        st.error("Model schema not available. `EXPECTED_COLUMNS` not found.")
    else:
        cols_ui = st.columns(4)
        for i, feature in enumerate(sim_cols):
            with cols_ui[i % 4]:
                sim_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=1.0,
                    format="%.6f",
                    key=f"sim_{feature}",
                )

    sim_run = st.button("▶ Run Simulation", disabled=not bool(sim_cols))

    if sim_run and sim_cols:
        try:
            sim_df = pd.DataFrame([[sim_values[c] for c in sim_cols]], columns=sim_cols)
            sim_result = predict_dataframe(sim_df, model_name=model_name).iloc[0].to_dict()
            st.session_state["sim_result"] = sim_result
            st.session_state["sim_model_name"] = model_name
        except Exception as e:
            st.error(f"Simulation failed: {e}")

if st.session_state.get("sim_result") is not None:
    sim_result = st.session_state["sim_result"]
    sim_model_used = st.session_state.get("sim_model_name", model_name)

    st.markdown(
        f"<div class='section-heading'>Simulation Output — {sim_model_used}</div>",
        unsafe_allow_html=True,
    )

    # Match dashboard styling with existing metric tiles
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value">{sim_result.get("predicted_class", "")}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{float(sim_result.get("confidence", 0.0)):.3f}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Alert Level</div>
                <div class="metric-value">{sim_result.get("alert_level", "")}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Uncertain</div>
                <div class="metric-value">{bool(sim_result.get("uncertain", False))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top3 = pd.DataFrame(
        [
            {"class": sim_result.get("top_1_class"), "probability": sim_result.get("top_1_prob")},
            {"class": sim_result.get("top_2_class"), "probability": sim_result.get("top_2_prob")},
            {"class": sim_result.get("top_3_class"), "probability": sim_result.get("top_3_prob")},
        ]
    )
    top3["probability"] = top3["probability"].astype(float)

    st.markdown("<div class='section-heading'>Top 3 Predicted Classes</div>", unsafe_allow_html=True)
    st.dataframe(top3, use_container_width=True)

# ─────────────────────────────────────────────
# CHUNKED PROCESSOR — supports 400 MB files
# ─────────────────────────────────────────────
CHUNK_SIZE = 50_000  # rows per chunk

def process_large_file(file, selected_model_name: str):
    """
    Stream-process a CSV in fixed-size chunks.
    Estimates total rows from file size for a realistic progress bar.
    """
    # Rough row estimate: sample 5 000 rows to get avg bytes/row
    file.seek(0)
    sample_bytes = file.read(512 * 1024)          # read 512 KB sample
    file.seek(0)

    sample_lines = sample_bytes.decode("utf-8", errors="replace").splitlines()
    if len(sample_lines) > 1:
        avg_bytes_per_row = len(sample_bytes) / max(len(sample_lines) - 1, 1)
        est_total_rows = max(int(file.size / avg_bytes_per_row), 1)
    else:
        est_total_rows = 1_000_000              # safe fallback

    est_chunks = max(est_total_rows // CHUNK_SIZE, 1)

    progress_bar   = st.progress(0.0)
    status_text    = st.empty()
    rows_processed = 0
    all_results    = []

    reader = pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False)

    for i, chunk in enumerate(reader):
        result = predict_dataframe(chunk, model_name=selected_model_name)
        all_results.append(result)

        rows_processed += len(chunk)
        pct = min((i + 1) / est_chunks, 0.99)      # cap at 99 % until done
        progress_bar.progress(pct)
        status_text.markdown(
            f"<span style='font-family:Share Tech Mono;font-size:0.85rem;color:#5e7a92'>"
            f"Processing chunk {i+1} &nbsp;·&nbsp; {rows_processed:,} rows analysed…"
            f"</span>",
            unsafe_allow_html=True,
        )

    progress_bar.progress(1.0)
    status_text.markdown(
        "<span style='font-family:Share Tech Mono;font-size:0.85rem;color:#00e5a0'>"
        "✔ All chunks processed</span>",
        unsafe_allow_html=True,
    )

    return pd.concat(all_results, ignore_index=True)

# ─────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Initialising detection engine…"):
        try:
            results = process_large_file(uploaded_file, selected_model_name=model_name)
            st.session_state.results = results
            st.success(f"✔ Detection complete — **{len(results):,}** rows analysed.")
        except Exception as e:
            st.error(f"Detection failed: {e}")

# ─────────────────────────────────────────────
# STEP 3 — Results
# ─────────────────────────────────────────────
st.markdown("<hr style='border-color:#1a2d42;margin:2rem 0;'>", unsafe_allow_html=True)

st.markdown("""
<div class="step-card">
    <div class="step-label">Step 03</div>
    <div class="step-title">Threat Analysis &amp; Results</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.results is not None:
    results = st.session_state.results

    high_alerts  = int((results["alert_level"] == "HIGH_ALERT").sum())
    uncertain    = int(results["uncertain"].sum())
    avg_conf     = results["confidence"].mean()
    total        = len(results)

    # ── Metric tiles ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="metric-label">Total Rows</div>
            <div class="metric-value ok">{total:,}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">High Alerts</div>
            <div class="metric-value alert">{high_alerts:,}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Uncertain</div>
            <div class="metric-value warn">{uncertain:,}</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{avg_conf:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts side by side ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-heading">Alert Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(results["alert_level"].value_counts())

    with col_b:
        st.markdown('<div class="section-heading">Top Attack Types</div>', unsafe_allow_html=True)
        st.bar_chart(results["predicted_class"].value_counts().head(10))

    # ── Sample data ──
    st.markdown('<div class="section-heading">Sample Results (first 20 rows)</div>', unsafe_allow_html=True)
    st.dataframe(results.head(20), use_container_width=True)

    # ── Download ──
    st.markdown("")
    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Download Full Results (CSV)",
        data=csv_data,
        file_name="ids_results.csv",
        mime="text/csv",
    )

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;color:#2a4257;
                font-family:Share Tech Mono,monospace;font-size:0.95rem;letter-spacing:0.1em;'>
        NO RESULTS YET — UPLOAD A FILE AND RUN DETECTION
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────
st.markdown("<hr style='border-color:#1a2d42;margin:2rem 0;'>", unsafe_allow_html=True)

if st.button("↺  RESET SESSION"):
    st.session_state.results = None
    st.session_state.file_loaded = False
    st.rerun()

# ─────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────
st.markdown("<hr style='border-color:#1a2d42;margin:2rem 0;'>", unsafe_allow_html=True)

st.markdown("""
<div class="step-card">
    <div class="step-label">Explainability</div>
    <div class="step-title">Model Feature Importance</div>
</div>
""", unsafe_allow_html=True)

if FEATURE_IMPORTANCE_PATH.exists():
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    col_fi1, col_fi2 = st.columns([1, 2])
    with col_fi1:
        st.dataframe(fi.head(10), use_container_width=True)
    with col_fi2:
        st.bar_chart(fi.set_index("feature")["importance"].head(10))
else:
    st.warning("Feature importance file not found at expected path.")
