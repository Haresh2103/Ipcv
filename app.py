"""
Smart Road – Pothole & Crack Detection System
Frontend: Streamlit  |  Algorithms: Laplacian + Morphological Skeleton
"""

import io
import time
import base64
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from detection_engine import analyze_image, generate_sample_road_image

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartRoad – Defect Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0f19;
    color: #c8d8e8;
  }
  .stApp { background: #0b0f19; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1624 0%, #0b1020 100%);
    border-right: 1px solid #1e2d45;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #4fc3f7;
  }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #112240 40%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      90deg, transparent, transparent 40px,
      rgba(79,195,247,0.03) 40px, rgba(79,195,247,0.03) 41px
    );
  }
  .hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #4fc3f7;
    letter-spacing: 2px;
    margin: 0;
    text-shadow: 0 0 30px rgba(79,195,247,0.4);
  }
  .hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #607d9b;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.3);
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4fc3f7;
    margin-right: 8px;
    margin-top: 12px;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 140px;
    background: #0f1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
  }
  .metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, #4fc3f7);
  }
  .metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #607d9b;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  .metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent, #4fc3f7);
    line-height: 1.1;
  }
  .metric-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #607d9b;
  }

  /* Severity badge */
  .sev-low      { --sev-color: #4caf50; }
  .sev-medium   { --sev-color: #ffb300; }
  .sev-high     { --sev-color: #ff7043; }
  .sev-critical { --sev-color: #e53935; }
  .severity-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,0,0,0.4);
    border: 2px solid var(--sev-color);
    border-radius: 8px;
    padding: 10px 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--sev-color);
    text-shadow: 0 0 16px var(--sev-color);
    box-shadow: 0 0 20px rgba(0,0,0,0.5), inset 0 0 20px rgba(0,0,0,0.3);
  }
  .severity-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--sev-color);
    box-shadow: 0 0 8px var(--sev-color);
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.3); }
  }

  /* Section headers */
  .section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #4fc3f7;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 24px 0 16px;
  }

  /* Step pills */
  .step-pill {
    display: inline-block;
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4fc3f7;
    margin-bottom: 8px;
  }

  /* Image containers */
  .img-frame {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    overflow: hidden;
    background: #060c18;
  }
  .img-caption {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #607d9b;
    text-align: center;
    padding: 6px;
    background: #0a1220;
    border-top: 1px solid #1e2d45;
  }

  /* Alert / info boxes */
  .info-box {
    background: rgba(79,195,247,0.06);
    border-left: 3px solid #4fc3f7;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #90b8d0;
    margin: 12px 0;
  }
  .warn-box {
    background: rgba(255,179,0,0.07);
    border-left: 3px solid #ffb300;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #c8a030;
    margin: 12px 0;
  }

  /* Progress bar override */
  .stProgress > div > div > div { background: #4fc3f7; }

  /* Sliders */
  .stSlider .st-bx { background: #4fc3f7; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0b0f19;
    border-bottom: 1px solid #1e3a5f;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    color: #607d9b;
    background: transparent;
    border-radius: 6px 6px 0 0;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(79,195,247,0.1) !important;
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7;
  }

  /* Hide default streamlit branding */
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── UTILITIES ─────────────────────────────────────────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def img_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(bgr))


def render_metric_card(label: str, value: str, unit: str = "", accent: str = "#4fc3f7"):
    st.markdown(f"""
    <div class="metric-card" style="--accent:{accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)


def render_image_frame(bgr: np.ndarray, caption: str):
    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
    st.image(img_to_pil(bgr), use_container_width=True)
    st.markdown(f'<div class="img-caption">{caption}</div></div>', unsafe_allow_html=True)


def severity_css_class(label: str) -> str:
    return f"sev-{label.lower()}"


def download_button_bytes(bgr: np.ndarray, filename: str, label: str):
    pil = img_to_pil(bgr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    st.download_button(label=label, data=buf.getvalue(),
                       file_name=filename, mime="image/png")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛣️ SmartRoad")
    st.markdown("---")

    st.markdown("#### 📂 Input Source")
    source = st.radio("", ["Upload Image", "Use Demo Image", "Webcam Snapshot"],
                      label_visibility="collapsed")

    uploaded_file = None
    demo_seed = 42
    if source == "Upload Image":
        uploaded_file = st.file_uploader("Upload road image",
                                         type=["jpg", "jpeg", "png", "bmp", "tif"],
                                         label_visibility="collapsed")
    elif source == "Use Demo Image":
        demo_seed = st.selectbox("Demo scene", [42, 7, 13, 99, 2025],
                                 format_func=lambda x: f"Scene #{x}")
    else:
        uploaded_file = st.camera_input("Take a photo", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### ⚙️ Detection Parameters")

    lap_ksize = st.select_slider(
        "Laplacian kernel size",
        options=[1, 3, 5, 7],
        value=5,
        help="Larger = captures coarser edges (better for deep potholes)"
    )
    lap_thresh = st.slider(
        "Laplacian threshold",
        min_value=0.02, max_value=0.40, value=0.12, step=0.01,
        help="Higher = only very sharp discontinuities flagged"
    )
    morph_k = st.select_slider(
        "Morphology kernel size",
        options=[2, 3, 4, 5],
        value=3,
        help="Affects crack thinning & region merging"
    )
    blur_sigma = st.slider(
        "Pre-blur sigma",
        min_value=0.5, max_value=4.0, value=1.5, step=0.25,
        help="More blur = fewer false positives from texture"
    )
    min_pothole = st.slider(
        "Min pothole area (px²)",
        min_value=100, max_value=2000, value=300, step=50
    )
    min_crack = st.slider(
        "Min crack area (px²)",
        min_value=20, max_value=500, value=60, step=10
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Algorithm Info")
    with st.expander("Laplacian Detection"):
        st.write("""
        Applies the 2nd-order Laplacian operator to the CLAHE-enhanced
        grayscale image. High absolute response indicates abrupt depth or
        texture changes — typical of pothole edges. Morphological closing
        merges fragments into blobs.
        """)
    with st.expander("Morphological Skeleton"):
        st.write("""
        Adaptive thresholding extracts dark linear features. The
        **skeletonize** algorithm (scikit-image) reduces these to
        single-pixel centrelines. This faithfully represents crack
        topology and allows accurate length/orientation analysis.
        """)


# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🛣️ SMARTROAD DETECT</div>
  <div class="hero-sub">Intelligent Road Surface Analysis Platform</div>
  <div>
    <span class="hero-badge">LAPLACIAN EDGE</span>
    <span class="hero-badge">MORPHOLOGICAL SKELETON</span>
    <span class="hero-badge">CLAHE PREPROCESSING</span>
    <span class="hero-badge">REAL-TIME ANALYSIS</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── LOAD IMAGE ─────────────────────────────────────────────────────────────────
image_bgr = None

if source == "Upload Image" and uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif source == "Use Demo Image":
    image_bgr = generate_sample_road_image(seed=demo_seed)
elif source == "Webcam Snapshot" and uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    # Landing state
    st.markdown("""
    <div class="info-box">
      📡 <strong>Ready for Analysis</strong> — Upload a road image, choose a demo scene,
      or capture with your camera using the sidebar panel.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="--accent:#4fc3f7">
          <div class="metric-label">Step 1</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;color:#c8d8e8;margin-top:6px;">
            Load Image
          </div>
          <div style="font-size:0.8rem;color:#607d9b;margin-top:4px;">
            Upload or pick a demo road scene from the sidebar
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="--accent:#ffb300">
          <div class="metric-label">Step 2</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;color:#c8d8e8;margin-top:6px;">
            Configure
          </div>
          <div style="font-size:0.8rem;color:#607d9b;margin-top:4px;">
            Tune Laplacian & skeleton params for your image
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="--accent:#4caf50">
          <div class="metric-label">Step 3</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;color:#c8d8e8;margin-top:6px;">
            Analyse
          </div>
          <div style="font-size:0.8rem;color:#607d9b;margin-top:4px;">
            Results appear instantly — severity, counts, heatmap
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ── RESIZE FOR PERFORMANCE ────────────────────────────────────────────────────
MAX_DIM = 960
h, w = image_bgr.shape[:2]
if max(h, w) > MAX_DIM:
    scale = MAX_DIM / max(h, w)
    image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)


# ── ANALYSIS ───────────────────────────────────────────────────────────────────
with st.spinner("🔬 Running Laplacian + Skeleton analysis…"):
    t0 = time.time()
    result = analyze_image(
        image_bgr,
        laplacian_ksize=lap_ksize,
        laplacian_threshold=lap_thresh,
        morph_kernel_size=morph_k,
        min_pothole_area=min_pothole,
        min_crack_area=min_crack,
        blur_sigma=blur_sigma,
    )
    elapsed = time.time() - t0


# ── SEVERITY BANNER ───────────────────────────────────────────────────────────
sev_colors = {
    "Low": "#4caf50", "Medium": "#ffb300", "High": "#ff7043", "Critical": "#e53935"
}
sev_color = sev_colors[result.severity_label]
sev_class = severity_css_class(result.severity_label)

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px">
  <div class="severity-badge {sev_class}">
    <div class="severity-dot" style="--sev-color:{sev_color}"></div>
    SEVERITY: {result.severity_label.upper()}
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#607d9b">
    Analysis completed in {elapsed*1000:.0f} ms
    &nbsp;|&nbsp; Resolution: {image_bgr.shape[1]}×{image_bgr.shape[0]} px
  </div>
</div>
""", unsafe_allow_html=True)


# ── METRICS ROW ───────────────────────────────────────────────────────────────
st.markdown('<div class="metric-row">', unsafe_allow_html=True)
cols = st.columns(5)
metrics = [
    ("Potholes", str(result.pothole_count), "detected", "#ff7043"),
    ("Cracks", str(result.crack_count), "detected", "#ffb300"),
    ("Defect Area", f"{result.total_defect_area_pct:.1f}", "% of surface", "#4fc3f7"),
    ("Severity Score", f"{result.severity_score:.0f}", "/ 100", sev_color),
    ("Process Time", f"{elapsed*1000:.0f}", "ms", "#4caf50"),
]
for col, (label, val, unit, acc) in zip(cols, metrics):
    with col:
        render_metric_card(label, val, unit, acc)
st.markdown('</div>', unsafe_allow_html=True)

# Severity progress bar
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#607d9b;white-space:nowrap">
    SEVERITY INDEX
  </div>
  <div style="flex:1;height:6px;background:#1e2d45;border-radius:3px;overflow:hidden">
    <div style="width:{result.severity_score}%;height:100%;background:{sev_color};
      border-radius:3px;box-shadow:0 0 8px {sev_color};transition:width 0.5s ease"></div>
  </div>
  <div style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;color:{sev_color};font-weight:700">
    {result.severity_score:.0f}%
  </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_results, tab_pipeline, tab_analysis, tab_report = st.tabs([
    "📊  Results", "🔬  Pipeline Steps", "📈  Analysis", "📋  Report"
])


# ── TAB 1: RESULTS ─────────────────────────────────────────────────────────────
with tab_results:
    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.markdown('<div class="section-header">ORIGINAL IMAGE</div>', unsafe_allow_html=True)
        render_image_frame(result.original, "Input frame")
    with col_annot:
        st.markdown('<div class="section-header">DETECTION OUTPUT</div>', unsafe_allow_html=True)
        render_image_frame(result.annotated,
                           "RED = Potholes  |  CYAN = Cracks")

    st.markdown("---")
    col_sev, col_mask = st.columns(2)
    with col_sev:
        st.markdown('<div class="section-header">SEVERITY HEATMAP</div>', unsafe_allow_html=True)
        render_image_frame(result.severity_map,
                           "Warm = high defect density")
    with col_mask:
        st.markdown('<div class="section-header">COMBINED DEFECT MASK</div>', unsafe_allow_html=True)
        # Colorize the binary mask
        mask_color = np.zeros((*result.defect_mask.shape, 3), dtype=np.uint8)
        mask_color[result.defect_mask > 0] = [0, 220, 255]
        render_image_frame(mask_color, "All detected defects")

    st.markdown("---")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        download_button_bytes(result.annotated, "detection_output.png", "⬇️ Download Annotated")
    with dl2:
        download_button_bytes(result.severity_map, "severity_heatmap.png", "⬇️ Download Heatmap")
    with dl3:
        download_button_bytes(mask_color, "defect_mask.png", "⬇️ Download Mask")


# ── TAB 2: PIPELINE STEPS ──────────────────────────────────────────────────────
with tab_pipeline:
    st.markdown("""
    <div class="info-box">
      Walk through each stage of the detection pipeline — from raw input to final annotations.
    </div>
    """, unsafe_allow_html=True)

    steps_meta = [
        ("1_preprocessed", "Step 1 — CLAHE Pre-processing",
         "Histogram equalisation (CLAHE) normalises contrast. Gaussian blur suppresses sensor noise before derivative operations."),
        ("2_laplacian",    "Step 2 — Laplacian Map (Pothole Detection)",
         "Second-order Laplacian filter highlights abrupt surface discontinuities. Bright regions indicate edges of potholes or severe damage. Thresholded and morphologically closed to form blobs."),
        ("3_skeleton",     "Step 3 — Morphological Skeleton (Crack Detection)",
         "Adaptive threshold extracts dark linear features. Skeletonize (Zhang-Suen) reduces crack regions to single-pixel centrelines, preserving topology."),
        ("4_pothole_mask", "Step 4a — Pothole Mask",
         "Filtered blobs (area & aspect ratio) from the Laplacian stage."),
        ("4_crack_mask",   "Step 4b — Crack Mask",
         "Elongated skeleton regions after subtracting pothole blobs."),
        ("5_annotated",    "Step 5 — Final Annotation",
         "Pothole bounding boxes (red) and crack overlays (cyan) drawn on the original image."),
    ]

    for key, title, desc in steps_meta:
        if key not in result.processing_steps:
            continue
        st.markdown(f'<div class="step-pill">{title}</div>', unsafe_allow_html=True)
        render_image_frame(result.processing_steps[key], desc)
        st.markdown(f"<p style='font-size:0.82rem;color:#607d9b;margin-top:4px'>{desc}</p>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ── TAB 3: ANALYSIS ────────────────────────────────────────────────────────────
with tab_analysis:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    col_l, col_r = st.columns(2)

    # Laplacian response histogram
    with col_l:
        st.markdown('<div class="section-header">LAPLACIAN RESPONSE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_eq, (0, 0), blur_sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_ksize)
        lap_norm = np.abs(lap) / (np.abs(lap).max() + 1e-8)

        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f1a2e")
        ax.set_facecolor("#0b1020")
        counts, bins, patches = ax.hist(lap_norm.ravel(), bins=80, color="#4fc3f7", alpha=0.8)
        ax.axvline(lap_thresh, color="#ff7043", linewidth=1.5, linestyle="--",
                   label=f"Threshold = {lap_thresh:.2f}")
        ax.fill_betweenx([0, counts.max()], lap_thresh, 1.0,
                         alpha=0.15, color="#ff7043")
        ax.set_xlabel("Normalised Laplacian magnitude", color="#607d9b", fontsize=8)
        ax.set_ylabel("Pixel count", color="#607d9b", fontsize=8)
        ax.tick_params(colors="#607d9b", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        ax.legend(fontsize=7, facecolor="#0f1a2e", edgecolor="#1e3a5f",
                  labelcolor="#c8d8e8")
        ax.set_title("Pothole candidate pixels (red shaded)", color="#90b8d0",
                     fontsize=8, pad=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Defect area breakdown pie
    with col_r:
        st.markdown('<div class="section-header">SURFACE CONDITION BREAKDOWN</div>',
                    unsafe_allow_html=True)
        pot_px = (result.processing_steps.get("4_pothole_mask",
                  np.zeros_like(image_bgr))[:, :, 0] > 0).sum()
        crack_px = (result.processing_steps.get("4_crack_mask",
                    np.zeros_like(image_bgr))[:, :, 0] > 0).sum()
        total_px = image_bgr.shape[0] * image_bgr.shape[1]
        healthy_px = total_px - pot_px - crack_px

        fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor="#0f1a2e")
        ax2.set_facecolor("#0b1020")
        sizes = [max(0, pot_px), max(0, crack_px), max(0, healthy_px)]
        labels = ["Potholes", "Cracks", "Intact Surface"]
        colors_pie = ["#ff7043", "#ffb300", "#37474f"]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=colors_pie,
            autopct="%1.1f%%", startangle=140,
            textprops={"color": "#c8d8e8", "fontsize": 8},
            wedgeprops={"edgecolor": "#0b1020", "linewidth": 2},
        )
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontsize(7)
        ax2.set_title("Pixel-level surface classification", color="#90b8d0",
                      fontsize=8, pad=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Crack skeleton length & orientation
    st.markdown('<div class="section-header">CRACK REGION PROPERTIES</div>',
                unsafe_allow_html=True)
    if result.crack_regions:
        import pandas as pd
        rows = []
        for i, prop in enumerate(result.crack_regions):
            rows.append({
                "ID": i + 1,
                "Area (px²)": int(prop.area),
                "Major axis (px)": round(prop.major_axis_length, 1),
                "Minor axis (px)": round(prop.minor_axis_length, 1),
                "Orientation (°)": round(np.degrees(prop.orientation), 1),
                "Extent": round(prop.extent, 3),
                "Eccentricity": round(prop.eccentricity, 3),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=220)
    else:
        st.markdown('<div class="info-box">No crack regions detected at current thresholds.</div>',
                    unsafe_allow_html=True)


# ── TAB 4: REPORT ─────────────────────────────────────────────────────────────
with tab_report:
    sev_action = {
        "Low": "Schedule routine maintenance within 6 months.",
        "Medium": "Plan maintenance within 30–60 days. Monitor progression.",
        "High": "Prioritise repair within 2 weeks. Consider temporary patching.",
        "Critical": "IMMEDIATE ACTION REQUIRED. Road may be hazardous. Close affected lane.",
    }

    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_md = f"""
## 🛣️ SmartRoad Defect Analysis Report

**Generated:** {now}  
**Algorithm:** Laplacian (ksize={lap_ksize}, thresh={lap_thresh}) + Morphological Skeleton (k={morph_k})  
**Resolution:** {image_bgr.shape[1]} × {image_bgr.shape[0]} px

---

### Executive Summary

| Metric | Value |
|---|---|
| **Severity Label** | {result.severity_label} |
| **Severity Score** | {result.severity_score:.1f} / 100 |
| **Potholes Detected** | {result.pothole_count} |
| **Cracks Detected** | {result.crack_count} |
| **Total Defect Area** | {result.total_defect_area_pct:.2f}% of surface |
| **Processing Time** | {elapsed*1000:.0f} ms |

---

### Recommended Action

> **{sev_action[result.severity_label]}**

---

### Technical Notes

- **Pre-processing:** CLAHE (clipLimit=2.5, tileGrid=8×8) + Gaussian blur σ={blur_sigma}
- **Pothole detection:** Laplacian 2nd-order derivative → threshold → morphological close
- **Crack detection:** Adaptive threshold → morphological open → Zhang-Suen skeletonize → dilate
- **Post-filtering:** Min pothole area = {min_pothole} px², Min crack area = {min_crack} px²

### Pothole Regions
{f"Detected {result.pothole_count} region(s) with combined area {sum(p.area for p in result.pothole_regions):,} px²" if result.pothole_regions else "None detected."}

### Crack Regions
{f"Detected {result.crack_count} region(s). Longest crack axis: {max((p.major_axis_length for p in result.crack_regions), default=0):.1f} px" if result.crack_regions else "None detected."}
"""

    st.markdown(report_md)
    st.download_button(
        "⬇️ Download Report (.md)",
        data=report_md,
        file_name="smartroad_report.md",
        mime="text/markdown"
    )


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:20px;
  border-top:1px solid #1e2d45;
  font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#2d4a6b">
  SMARTROAD DETECT v1.0 &nbsp;|&nbsp;
  Laplacian + Morphological Skeleton Pipeline &nbsp;|&nbsp;
  Powered by OpenCV · scikit-image · Streamlit
</div>
""", unsafe_allow_html=True)
