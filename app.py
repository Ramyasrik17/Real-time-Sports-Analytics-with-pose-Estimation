"""
app.py
Streamlit dashboard - the frontend. Professional styling: custom CSS,
gauge-style score meter, color-coded status cards, polished layout.

Run with:
    streamlit run app.py
"""

import json
import os
import tempfile
import time

import cv2
import mediapipe as mp
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pose_utils import extract_joint_angles, compute_deviation_score
import db

BASELINE_DIR = "baselines"

st.set_page_config(
    page_title="Sports Analytics | Pose Estimation",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)
db.init_db()

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background-color: #0E1117;
    }

    /* Hero header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1F4E79 0%, #2E86AB 50%, #1B998B 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .hero-title {
        color: #FFFFFF;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 15px;
        margin-top: 6px;
        font-weight: 400;
    }

    /* KPI cards */
    .kpi-card {
        background: #161B22;
        border: 1px solid #2A2F3A;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .kpi-label {
        color: #8B949E;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        margin: 0;
    }
    .kpi-good { color: #3FB950; }
    .kpi-warn { color: #D29922; }
    .kpi-bad  { color: #F85149; }
    .kpi-neutral { color: #58A6FF; }

    /* Section card wrapper */
    .section-card {
        background: #161B22;
        border: 1px solid #2A2F3A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .section-title {
        color: #E6EDF3;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 12px;
        border-left: 4px solid #2E86AB;
        padding-left: 10px;
    }

    /* Badge pills for joint flags */
    .badge-ok {
        background: rgba(63,185,80,0.15);
        color: #3FB950;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-flag {
        background: rgba(248,81,73,0.15);
        color: #F85149;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #2A2F3A;
    }

    /* Hide default streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### ⚙️ Session Setup")
athlete_name = st.sidebar.text_input("Athlete name", value="athlete1")
tolerance_deg = st.sidebar.slider("Deviation tolerance (°)", 5, 45, 15)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎥 Input Source")
input_source = st.sidebar.radio(
    "Choose input",
    ["Live Camera", "Upload Video"],
    label_visibility="collapsed",
)

camera_index = 0
uploaded_video_path = None
run_session = False

if input_source == "Live Camera":
    camera_index = st.sidebar.number_input("Camera index", value=0, min_value=0, step=1)
    run_session = st.sidebar.toggle("▶️ Start camera")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file", type=["mp4", "mov", "avi", "mkv"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        uploaded_video_path = tfile.name
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        run_session = st.sidebar.toggle("▶️ Start analysis")
    else:
        st.sidebar.caption("Upload a video file to enable analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📌 Quick Guide")
st.sidebar.caption(
    "1. Record a baseline first:\n"
    "`python baseline_recorder.py --name <name>`\n\n"
    "2. Enter the same name above.\n\n"
    "3. Choose Live Camera or Upload Video, then start."
)

# ============================================================
# LOAD BASELINE
# ============================================================
baseline_path = os.path.join(BASELINE_DIR, f"{athlete_name}.json")
baseline = None
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🏃 Real-Time Sports Analytics</p>
    <p class="hero-subtitle">Pose Estimation &nbsp;→&nbsp; Joint Angle Calculation &nbsp;→&nbsp; Form Deviation Detection &nbsp;→&nbsp; Live Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# BASELINE STATUS BANNER
# ============================================================
if baseline is None:
    st.warning(f"⚠️ No baseline found for **{athlete_name}**. Detection will run, but scoring stays disabled until a baseline is recorded.")
else:
    st.success(f"✅ Baseline loaded for **{athlete_name}** — live deviation scoring is active.")

# ============================================================
# GAUGE CHART HELPER
# ============================================================
def render_gauge(score):
    if score >= 80:
        bar_color = "#3FB950"
    elif score >= 50:
        bar_color = "#D29922"
    else:
        bar_color = "#F85149"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 36, "color": "#E6EDF3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#0E1117",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(248,81,73,0.15)"},
                {"range": [50, 80], "color": "rgba(210,153,34,0.15)"},
                {"range": [80, 100], "color": "rgba(63,185,80,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6EDF3"},
    )
    return fig


def kpi_card(label, value, css_class):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT
# ============================================================
col_video, col_metrics = st.columns([1.6, 1])

with col_video:
    st.markdown('<div class="section-title">📹 Live Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

with col_metrics:
    st.markdown('<div class="section-title">📊 Live Metrics</div>', unsafe_allow_html=True)
    gauge_placeholder = st.empty()
    kpi_row_placeholder = st.empty()
    table_placeholder = st.empty()

chart_placeholder = st.empty()
history_placeholder = st.container()

# ============================================================
# LIVE / VIDEO LOOP
# ============================================================
if run_session and (input_source == "Live Camera" or uploaded_video_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if input_source == "Live Camera":
        cap = cv2.VideoCapture(camera_index)
        source_label = "webcam"
    else:
        cap = cv2.VideoCapture(uploaded_video_path)
        source_label = "uploaded video"

    if not cap.isOpened():
        st.error(f"❌ Could not open {source_label}. Check the source and try again.")
    else:
        score_log = []
        frame_count = 0
        max_frames = 100000
        deviations = {}

        while run_session and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                if input_source == "Upload Video":
                    st.info("📹 Reached end of uploaded video.")
                else:
                    st.error("Lost connection to webcam.")
                break

            if input_source == "Live Camera":
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            current_score = None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(46, 134, 171), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(27, 153, 139), thickness=2),
                )
                h, w = frame.shape[:2]
                angles = extract_joint_angles(results.pose_landmarks.landmark, w, h)

                if baseline:
                    current_score, deviations = compute_deviation_score(angles, baseline, tolerance_deg)
                    score_log.append(current_score)
            else:
                cv2.putText(frame, "No person detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            if current_score is not None:
                gauge_placeholder.plotly_chart(render_gauge(current_score), use_container_width=True, key=f"gauge_{frame_count}")

                flagged_count = sum(1 for info in deviations.values() if info["flagged"])
                status_class = "kpi-good" if flagged_count == 0 else ("kpi-warn" if flagged_count <= 2 else "kpi-bad")
                status_text = "Good Form" if flagged_count == 0 else f"{flagged_count} Joint(s) Off"

                with kpi_row_placeholder.container():
                    k1, k2 = st.columns(2)
                    with k1:
                        kpi_card("Status", status_text, status_class)
                    with k2:
                        kpi_card("Frames Logged", len(score_log), "kpi-neutral")

                rows = []
                for j, info in deviations.items():
                    badge = '<span class="badge-flag">FLAGGED</span>' if info["flagged"] else '<span class="badge-ok">OK</span>'
                    rows.append({
                        "Joint": j.replace("_", " ").title(),
                        "Current (°)": info["current"],
                        "Baseline (°)": info["baseline"],
                        "Δ (°)": info["deviation"],
                        "Status": badge,
                    })
                df_display = pd.DataFrame(rows)
                table_placeholder.markdown(
                    df_display.to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )

                if len(score_log) > 1:
                    trend_fig = go.Figure()
                    trend_fig.add_trace(go.Scatter(
                        y=score_log, mode="lines", fill="tozeroy",
                        line=dict(color="#2E86AB", width=2),
                        fillcolor="rgba(46,134,171,0.15)",
                    ))
                    trend_fig.update_layout(
                        title="Form Score Trend (this session)",
                        height=220,
                        margin=dict(l=20, r=20, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#E6EDF3"},
                        yaxis=dict(range=[0, 100], gridcolor="#2A2F3A"),
                        xaxis=dict(gridcolor="#2A2F3A"),
                    )
                    chart_placeholder.plotly_chart(trend_fig, use_container_width=True, key=f"trend_{frame_count}")
            else:
                gauge_placeholder.info("Detecting pose... (no baseline loaded)")

            frame_count += 1
            time.sleep(0.01)

        cap.release()
        pose.close()

        if score_log:
            avg_score = round(sum(score_log) / len(score_log), 1)
            flagged_joints = ",".join(j for j, info in deviations.items() if info.get("flagged"))
            db.save_session(athlete_name, avg_score, flagged_joints)
            st.success(f"✅ Session saved — average score: **{avg_score}/100**")

# ============================================================
# SESSION HISTORY
# ============================================================
with history_placeholder:
    st.markdown("---")
    st.markdown(f'<div class="section-title">🗂️ Session History — {athlete_name}</div>', unsafe_allow_html=True)

    sessions = db.get_sessions(athlete_name)
    if sessions:
        df = pd.DataFrame(sessions)

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Total Sessions", len(df), "kpi-neutral")
        with c2:
            kpi_card("Best Score", f"{df['avg_score'].max():.1f}", "kpi-good")
        with c3:
            kpi_card("Average Score", f"{df['avg_score'].mean():.1f}", "kpi-warn")

        st.markdown("<br>", unsafe_allow_html=True)

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["avg_score"], mode="lines+markers",
            line=dict(color="#1B998B", width=2),
            marker=dict(size=8, color="#2E86AB"),
        ))
        hist_fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E6EDF3"},
            yaxis=dict(range=[0, 100], title="Score", gridcolor="#2A2F3A"),
            xaxis=dict(title="Session", gridcolor="#2A2F3A"),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        st.dataframe(
            df[["timestamp", "avg_score", "flagged_joints"]].rename(columns={
                "timestamp": "Date/Time", "avg_score": "Avg Score", "flagged_joints": "Flagged Joints"
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No past sessions yet. Complete a session above to start building your history.")