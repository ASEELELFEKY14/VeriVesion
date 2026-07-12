import os
import sys
import base64
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLASS_LABELS,
    EXTRACTED_FRAMES_DISPLAY_COUNT,
    MODEL_PATH,
    OUTPUT_DIR,
    POSITIVE_CLASS,
)
from src.model import load_model
from src.predictor import analyze_image
from src.report_generator import (
    generate_image_pdf_report,
    generate_video_pdf_report,
    get_clinical_recommendation,
    get_medical_interpretation,
)
from src.video_converter import ensure_compatible_video
from src.video_processor import extract_frames


st.set_page_config(
    page_title="Endoscopy Diagnostic System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root{
        --navy:#06264a;--navy2:#031d3a;--blue:#0b6ff0;--green:#18b884;
        --orange:#ff7a00;--red:#ff4b5c;--text:#17243a;--muted:#64748b;
        --line:#dfe7f1;--bg:#f5f8fc;
    }
    html,body,[class*="css"]{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;}
    .stApp{background:var(--bg);color:var(--text);}
    header[data-testid="stHeader"],[data-testid="collapsedControl"]{display:none;}
    .block-container{padding:0 26px 30px 26px;max-width:1400px;}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,var(--navy),var(--navy2));min-width:240px!important;max-width:240px!important;}
    [data-testid="stSidebar"]>div{padding:22px 14px;}
    .brand{text-align:center;color:white;padding:4px 0 26px;}
    .logo{width:78px;height:78px;margin:0 auto 14px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#28a8ff,#0878ef 50%,#04458f);display:flex;align-items:center;justify-content:center;box-shadow:0 14px 28px rgba(0,0,0,.25);}
    .brand h2{margin:0;font-size:20px;line-height:1.1;font-weight:850;}
    .brand p{margin:5px 0 0;color:rgba(255,255,255,.78);font-size:14px;}
    .nav{height:48px;border-radius:8px;display:flex;align-items:center;gap:13px;padding:0 14px;color:rgba(255,255,255,.9);font-weight:750;margin-bottom:8px;font-size:15px;}
    .nav.active{background:var(--blue);color:white;box-shadow:0 10px 20px rgba(0,91,214,.25);}
    .nav i{font-style:normal;width:20px;text-align:center;font-size:18px;}
    .light{margin-top:245px;background:rgba(14,102,190,.35);height:46px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:center;gap:10px;font-weight:800;}
    .topbar{height:82px;background:white;border-bottom:1px solid var(--line);margin:0 -26px 20px;padding:0 26px;display:flex;align-items:center;justify-content:space-between;}
    .topbar h1{margin:0;font-size:24px;font-weight:900;letter-spacing:0;}
    .topbar p{margin:5px 0 0;color:var(--muted);font-size:14px;font-weight:600;}
    .profile{display:flex;align-items:center;gap:16px;}
    .bell{position:relative;font-size:24px}.badge{position:absolute;top:-4px;right:-7px;background:#ff2d3d;color:white;border-radius:50%;width:16px;height:16px;font-size:10px;display:flex;align-items:center;justify-content:center;font-weight:900;}
    .avatar{width:42px;height:42px;border-radius:50%;background:linear-gradient(135deg,#0b6ff0,#18b884);color:white;display:flex;align-items:center;justify-content:center;font-weight:900;box-shadow:inset 0 0 0 3px #edf5ff;}
    .profile strong{display:block;font-size:14px}.profile span{font-size:12px;color:var(--muted);font-weight:700;}
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin-bottom:16px;}
    .stat {background:white;border:1px solid var(--line);border-radius:8px;min-height:108px;padding:18px;display:flex;align-items:center;gap:18px;box-shadow:0 4px 12px rgba(15,23,42,.04);}
    .stat-icon{width:58px;height:58px;border-radius:8px;color:white;display:flex;align-items:center;justify-content:center;font-size:28px;flex:0 0 auto;}
    .b{background:var(--blue)}.g{background:var(--green)}.o{background:var(--orange)}.r{background:var(--red)}
    .stat span{display:block;color:var(--muted);font-size:13px;font-weight:850;margin-bottom:7px;}
    .stat strong{display:block;font-size:24px;line-height:1;font-weight:950;margin-bottom:10px;}
    .stat small{color:#48566c;font-size:12px;font-weight:750;}
    .panel{background:white;border:1px solid var(--line);border-radius:8px;padding:18px;box-shadow:0 4px 12px rgba(15,23,42,.04);margin-bottom:16px;}
    .title{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:900;margin-bottom:16px;}
    .title i{font-style:normal;color:var(--blue);font-size:20px;}
    .exam{display:grid;grid-template-columns:280px minmax(300px,1fr) 380px;gap:22px;align-items:stretch;}
    .row{display:grid;grid-template-columns:26px 1fr;gap:12px;margin-bottom:18px;font-size:14px;font-weight:700;}
    .row i{font-style:normal;color:var(--blue);text-align:center}.row span{display:block;color:var(--muted);font-size:13px;margin-bottom:4px;font-weight:800;}
    .preview{height:292px;border-radius:8px;overflow:hidden;background:#07111f;border:1px solid #07111f;display:flex;align-items:center;justify-content:center;color:#90a4bd;font-weight:850;}
    .preview img{width:100%;height:100%;object-fit:cover;}
    .diag{background:linear-gradient(135deg,#ff3548,#ef2e3a);color:white;border-radius:8px;padding:18px 20px;display:flex;align-items:center;gap:14px;margin-bottom:18px;box-shadow:0 12px 24px rgba(239,46,58,.2);}
    .diag.neg{background:linear-gradient(135deg,#18b884,#0a9d70);box-shadow:0 12px 24px rgba(24,184,132,.2);}
    .diag-icon{width:40px;height:40px;border:2px solid rgba(255,255,255,.8);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:900;}
    .diag strong{display:block;font-size:22px;line-height:1;margin-bottom:5px}.diag span{font-size:14px;font-weight:800;}
    .conf{display:grid;grid-template-columns:135px 1fr;gap:18px;margin-bottom:16px;}
    .label{color:#45546b;font-size:13px;font-weight:900;margin-bottom:10px;}
    .circle{width:112px;height:112px;border-radius:50%;background:conic-gradient(var(--blue) var(--score),#e9eff8 0);display:flex;align-items:center;justify-content:center;margin-left:5px;}
    .inner{width:82px;height:82px;border-radius:50%;background:white;display:flex;align-items:center;justify-content:center;font-size:21px;font-weight:950;}
    .ai{border-left:1px solid var(--line);padding-left:22px;font-size:14px;line-height:1.55;font-weight:700;color:#26364f;min-height:128px;}
    .rec{font-size:13px;font-weight:750;line-height:1.45;color:#24344d;}
    .summary{background:#eaf8f0;border:1px solid #d5efe0;border-radius:8px;padding:16px;font-size:13px;margin-bottom:20px;}
    .sumrow{display:grid;grid-template-columns:145px 1fr;gap:8px;margin-bottom:10px}.sumrow span{color:#44536a;font-weight:800}.sumrow strong{font-weight:950;}
    .download{width:100%;min-height:48px;border-radius:8px;background:var(--blue);color:white!important;display:flex;align-items:center;justify-content:center;text-decoration:none!important;font-weight:900;margin-bottom:14px;box-shadow:0 10px 18px rgba(11,111,240,.2);}
    .ghost{min-height:48px;border-radius:8px;border:1px solid var(--line);background:white;color:var(--blue);display:flex;align-items:center;justify-content:center;font-weight:900;}
    .empty{border:1px dashed #bfd0e6;border-radius:8px;min-height:240px;background:#f9fbff;display:flex;align-items:center;justify-content:center;color:var(--muted);font-weight:850;}
    .image-card{border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:14px;background:white;}
    .image-card-grid{display:grid;grid-template-columns:120px 1fr 240px;gap:18px;align-items:center;}
    .image-card img{width:120px;height:90px;object-fit:cover;border-radius:8px;}
    .image-index{font-size:13px;color:var(--muted);font-weight:800;margin-bottom:6px;}
    .image-name{font-size:16px;font-weight:900;color:var(--text);margin-bottom:5px;}
    .image-meta{font-size:13px;color:var(--muted);font-weight:700;}
    div[data-testid="stImage"] img{border-radius:8px;object-fit:cover;}
    .stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:8px!important;background:var(--blue)!important;border:1px solid var(--blue)!important;color:white!important;min-height:48px;font-weight:900!important;}
    [data-testid="stFileUploader"] section{border:1px dashed #bfd0e6;border-radius:8px;background:#f9fbff;}
    .meta{text-align:center;color:#25364e;font-size:12px;font-weight:800;margin-top:4px;}
    @media(max-width:1150px){.stats,.exam,.image-card-grid{grid-template-columns:1fr}.conf{grid-template-columns:1fr}.ai{border-left:0;padding-left:0}}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    defaults = {
        "patient_registered": False,
        "patient_name": "",
        "patient_phone": "",
        "analysis_type": "Image",
        "image_result": None,
        "image_results": [],
        "video_result": None,
        "report_path": None,
        "report_paths": [],
        "examination_time": None,
        "uploaded_file_name": None,
        "selected_image_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner="Loading AI model weights...")
def get_model():
    return load_model(MODEL_PATH)


class VideoReportData:
    def __init__(self, final_class, final_label, confidence, total_frames, positive_frames, key_frames):
        self.final_class = final_class
        self.final_label = final_label
        self.confidence = confidence
        self.total_frames = total_frames
        self.positive_frames = positive_frames
        self.key_frames = key_frames


def image_to_base64(image_rgb):
    if image_rgb is None:
        return ""
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR) if image_rgb.ndim == 3 else image_rgb
    ok, buffer = cv.imencode(".jpg", image_bgr)
    return base64.b64encode(buffer).decode("utf-8") if ok else ""


def is_positive(predicted_class):
    return predicted_class == POSITIVE_CLASS


def display_confidence(predicted_class, score):
    return score * 100 if predicted_class == POSITIVE_CLASS else (1.0 - score) * 100


def get_selected_image_entry():
    image_results = st.session_state.get("image_results", [])
    if not image_results:
        return None

    selected_index = st.session_state.get("selected_image_index", 0)
    selected_index = max(0, min(selected_index, len(image_results) - 1))
    st.session_state["selected_image_index"] = selected_index
    return image_results[selected_index]


def reset_examination():
    st.session_state["image_result"] = None
    st.session_state["image_results"] = []
    st.session_state["video_result"] = None
    st.session_state["report_path"] = None
    st.session_state["report_paths"] = []
    st.session_state["examination_time"] = None
    st.session_state["uploaded_file_name"] = None
    st.session_state["selected_image_index"] = 0
    st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="logo">
                    <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M28 8C24.7 8 22 10.7 22 14V25.5C22 29.1 19.1 32 15.5 32H13C9.7 32 7 34.7 7 38C7 48.5 15.5 57 26 57H35C47.2 57 57 47.2 57 35C57 23.4 48.8 14.5 38.1 14.5H35.2C32.6 14.5 30.5 12.4 30.5 9.8C30.5 8.8 29.4 8 28 8Z" fill="white"/>
                        <path d="M24 18C24 16.9 24.9 16 26 16C27.1 16 28 16.9 28 18V26C28 32.1 23.1 37 17 37H14C12.9 37 12 37.9 12 39C12 46.7 18.3 53 26 53H35C45 53 53 45 53 35C53 25.8 46.7 18.5 38.1 18.5H35.2C29.9 18.5 25.5 14.6 24.8 9.5C24.3 10.5 24 11.7 24 13V18Z" fill="#eaf6ff"/>
                    </svg>
                </div>
                <h2>Endoscopy</h2>
                <p>Diagnostic System</p>
            </div>
            <div class="nav active"><i>⌂</i><span>Dashboard</span></div>
            <div class="nav"><i>＋</i><span>New Examination</span></div>
            <div class="nav"><i>◉</i><span>Patients</span></div>
            <div class="nav"><i>▤</i><span>Reports</span></div>
            <div class="nav"><i>▥</i><span>Statistics</span></div>
            <div class="nav"><i>⚙</i><span>Settings</span></div>
            <div class="nav"><i>♙</i><span>Users</span></div>
            <div class="nav"><i>?</i><span>Help & Support</span></div>
            <div class="light"><span>☼</span><span>Light Mode</span></div>
            """,
            unsafe_allow_html=True,
        )


def render_header():
    name = st.session_state["patient_name"] or "Patient"
    initials = "".join(part[:1] for part in name.split()[:2]).upper() or "P"
    st.markdown(
        f"""
        <div class="topbar">
            <div><h1>Endoscopy Diagnostic System</h1><p>AI-Powered Gastrointestinal Disease Detection</p></div>
            <div class="profile">
                <div class="bell">♧<span class="badge">3</span></div>
                <div class="avatar">{initials}</div>
                <div><strong>{name}</strong><span>Current patient</span></div>
                <div>⌄</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats():
    pos = neg = 0

    if st.session_state.get("image_results"):
        pos = sum(1 for item in st.session_state["image_results"] if is_positive(item["result"].predicted_class))
        neg = len(st.session_state["image_results"]) - pos
    elif st.session_state["video_result"] is not None:
        pos = 1 if is_positive(st.session_state["video_result"]["decision"]) else 0
        neg = 0 if pos else 1

    today = len(st.session_state.get("image_results", [])) if st.session_state.get("image_results") else (1 if st.session_state["patient_registered"] else 0)

    st.markdown(
        f"""
        <div class="stats">
            <div class="stat"><div class="stat-icon b">♙</div><div><span>Total Patients</span><strong>1,248</strong><small>All Patients ♙</small></div></div>
            <div class="stat"><div class="stat-icon g">▣</div><div><span>Today's Examinations</span><strong>{today}</strong><small>Today ▣</small></div></div>
            <div class="stat"><div class="stat-icon o">◎</div><div><span>Positive Cases</span><strong>{pos}</strong><small>This Week ↗</small></div></div>
            <div class="stat"><div class="stat-icon r">◇</div><div><span>Negative Cases</span><strong>{neg}</strong><small>This Week ↗</small></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_form():
    if st.session_state["patient_registered"]:
        return True

    st.markdown('<div class="panel"><div class="title"><i>♙</i>Current Examination</div>', unsafe_allow_html=True)
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="e.g. Mohamed Ali")
        with col2:
            patient_phone = st.text_input("Patient ID / Phone", placeholder="e.g. P-2024-1075")
        submitted = st.form_submit_button("Start Examination", use_container_width=True)

    if submitted:
        if not patient_name.strip() or not patient_phone.strip():
            st.error("Both fields are required.")
        else:
            st.session_state["patient_name"] = patient_name.strip()
            st.session_state["patient_phone"] = patient_phone.strip()
            st.session_state["patient_registered"] = True
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return False


def run_image_analysis(uploaded_files):
    try:
        if not uploaded_files:
            st.error("Please upload at least one image.")
            return

        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        model, device = get_model()
        image_results = []
        report_paths = []
        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            uploaded_file.seek(0)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

            if image_bgr is None:
                st.warning(f"Skipped unreadable image: {uploaded_file.name}")
                continue

            with st.spinner(f"Analyzing image {index + 1} of {len(uploaded_files)}..."):
                analysis_result = analyze_image(model, device, image_bgr)

            examination_time = datetime.now()
            report_path = OUTPUT_DIR / "reports" / (
                f"image_report_{index + 1}_{examination_time.strftime('%Y%m%d_%H%M%S_%f')}.pdf"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)

            generate_image_pdf_report(
                result=analysis_result,
                patient_name=st.session_state["patient_name"],
                patient_phone=st.session_state["patient_phone"],
                image_name=uploaded_file.name,
                output_path=report_path,
                examination_time=examination_time,
            )

            
            analysis_result.timestamp_sec = index * 120 + 15.0

            image_results.append(
                {
                    "result": analysis_result,
                    "file_name": uploaded_file.name,
                    "report_path": report_path,
                    "examination_time": examination_time,
                }
            )
            report_paths.append(report_path)
            progress.progress((index + 1) / len(uploaded_files))

        if not image_results:
            st.error("No images could be analyzed.")
            return

        st.session_state["image_results"] = image_results
        st.session_state["image_result"] = None
        st.session_state["video_result"] = None
        st.session_state["report_paths"] = report_paths
        st.session_state["report_path"] = report_paths[0]
        st.session_state["examination_time"] = image_results[0]["examination_time"]
        st.session_state["uploaded_file_name"] = ", ".join(item["file_name"] for item in image_results)
        st.session_state["selected_image_index"] = 0
        st.rerun()

    except Exception as e:
        st.error(f"Image Pipeline Interrupted: {e}")


def run_video_analysis(uploaded_file):
    raw_video_path = None
    try:
        temp_dir = OUTPUT_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        raw_video_path = temp_dir / uploaded_file.name

        with open(raw_video_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Optimizing video stream properties..."):
            compatible_video_path, _ = ensure_compatible_video(raw_video_path, temp_dir)

        model, device = get_model()

        with st.spinner("Selecting the most diagnostic video frames and running AI analysis..."):
            pipeline_result = extract_frames(
                str(compatible_video_path),
                model,
                device,
                top_k=EXTRACTED_FRAMES_DISPLAY_COUNT,
            )

        if not pipeline_result:
            st.error("No suitable diagnostic frames were found.")
            return

        examination_time = datetime.now()
        report_path = OUTPUT_DIR / "reports" / f"video_report_{examination_time.strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        final_class = pipeline_result["decision"]
        final_label = CLASS_LABELS[final_class]
        confidence_score = pipeline_result["decision_score"]
        extracted_frames = pipeline_result["extracted_frames"]

        key_frames = [f for f in extracted_frames if f.predicted_class == POSITIVE_CLASS][:5] or extracted_frames[:5]

        video_summary_data = VideoReportData(
            final_class=final_class,
            final_label=final_label,
            confidence=confidence_score,
            total_frames=len(extracted_frames),
            positive_frames=sum(1 for f in extracted_frames if f.predicted_class == POSITIVE_CLASS),
            key_frames=key_frames,
        )

        generate_video_pdf_report(
            result=video_summary_data,
            patient_name=st.session_state["patient_name"],
            patient_phone=st.session_state["patient_phone"],
            video_name=uploaded_file.name,
            output_path=report_path,
            examination_time=examination_time,
        )

        st.session_state["video_result"] = pipeline_result
        st.session_state["image_result"] = None
        st.session_state["image_results"] = []
        st.session_state["report_paths"] = []
        st.session_state["report_path"] = report_path
        st.session_state["examination_time"] = examination_time
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["selected_image_index"] = 0
        st.rerun()

    except Exception as e:
        st.error(f"Video Pipeline Interrupted: {e}")

    finally:
        if raw_video_path is not None and raw_video_path.exists():
            os.remove(raw_video_path)


def render_upload_area():
    st.markdown('<div class="panel"><div class="title"><i>＋</i>New Examination</div>', unsafe_allow_html=True)
    left, right = st.columns([0.44, 0.56], gap="large")

    with left:
        analysis_type = st.radio("Input Type", ["Image", "Video"], horizontal=True)
        st.session_state["analysis_type"] = analysis_type

        if analysis_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload Endoscopy Images",
                type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
                accept_multiple_files=True,
            )
        else:
            uploaded_file = st.file_uploader("Upload Endoscopy Video", type=None)

    with right:
        if not uploaded_file:
            st.markdown('<div class="empty">Waiting for examination media</div>', unsafe_allow_html=True)
        elif analysis_type == "Image":
            st.write(f"{len(uploaded_file)} image(s) selected")
            preview_cols = st.columns(min(len(uploaded_file), 3))
            for index, file in enumerate(uploaded_file[:3]):
                with preview_cols[index % 3]:
                    st.image(file, caption=file.name, use_container_width=True)
            if len(uploaded_file) > 3:
                st.caption(f"+ {len(uploaded_file) - 3} more image(s)")
        else:
            st.video(uploaded_file)

    if uploaded_file:
        if analysis_type == "Image":
            if st.button("Run Images Diagnosis", use_container_width=True):
                run_image_analysis(uploaded_file)
        else:
            if st.button("Run Video Diagnosis", use_container_width=True):
                run_video_analysis(uploaded_file)

    st.markdown("</div>", unsafe_allow_html=True)


def current_result_summary():
    # في حالة معالجة صور فردية متسلسلة
    if st.session_state.get("image_results"):
        results_list = st.session_state["image_results"]
        # لتحديد الفئة النهائية للتشخيص بناءً على ما إذا كانت هناك أي صورة إيجابية في المجموعة
        has_positive = any(is_positive(item["result"].predicted_class) for item in results_list)
        final_class = POSITIVE_CLASS if has_positive else results_list[0]["result"].predicted_class
        
        # تجميع درجات الثقة
        scores = [item["result"].probability for item in results_list]
        avg_score = sum(scores) / len(scores)
        confidence = display_confidence(final_class, avg_score)
        
        selected_index = st.session_state.get("selected_image_index", 0)
        selected_index = max(0, min(selected_index, len(results_list) - 1))
        active_res = results_list[selected_index]["result"]
        
        preview = active_res.roi_rgb if active_res.roi_rgb is not None else cv.cvtColor(active_res.image_bgr, cv.COLOR_BGR2RGB)
        positives = sum(1 for item in results_list if is_positive(item["result"].predicted_class))
        return final_class, confidence, preview, len(results_list), positives

    # في حالة معالجة الفيديو كالمعتاد
    if st.session_state["video_result"] is not None:
        pipeline = st.session_state["video_result"]
        final_class = pipeline["decision"]
        confidence = display_confidence(final_class, pipeline["decision_score"])
        frames = pipeline["extracted_frames"]
        preview = None
        if frames:
            preview = frames[0].roi_rgb if frames[0].roi_rgb is not None else cv.cvtColor(frames[0].image_bgr, cv.COLOR_BGR2RGB)
        positives = sum(1 for f in frames if f.predicted_class == POSITIVE_CLASS)
        return final_class, confidence, preview, len(frames), positives

    return None, None, None, 0, 0


def render_exam_overview():
    final_class, confidence, preview, total_frames, positive_frames = current_result_summary()
    has_result = final_class is not None
    exam_time = st.session_state["examination_time"] or datetime.now()
    display_file_name = st.session_state["uploaded_file_name"] or "No file selected"

    if has_result:
        label = CLASS_LABELS[final_class]
        verdict = "Positive" if is_positive(final_class) else "Negative"
        diag_class = "" if is_positive(final_class) else "neg"
        score = min(max(confidence, 0), 100)
        interpretation = get_medical_interpretation(final_class)
        recommendation = get_clinical_recommendation(final_class)
        img64 = image_to_base64(preview)
    else:
        label = "Awaiting Analysis"
        verdict = "Pending"
        diag_class = "neg"
        score = 0
        interpretation = "Upload endoscopy image(s) or video, then run the diagnosis to show the AI interpretation here."
        recommendation = "Clinical recommendation will appear after analysis."
        img64 = ""

    image_html = f'<img src="data:image/jpeg;base64,{img64}">' if img64 else "No preview yet"

    st.markdown(
        f"""
        <div class="panel">
            <div class="exam">
                <div>
                    <div class="title"><i>♙</i>Current Examination</div>
                    <div class="row"><i>♙</i><div><span>Patient Name</span>{st.session_state["patient_name"]}</div></div>
                    <div class="row"><i>▣</i><div><span>Patient ID</span>{st.session_state["patient_phone"]}</div></div>
                    <div class="row"><i>◎</i><div><span>Age / Gender</span>Not specified</div></div>
                    <div class="row"><i>▣</i><div><span>Date</span>{exam_time.strftime('%d %B %Y - %I:%M %p')}</div></div>
                    <div class="row"><i>▤</i><div><span>Video / Image Name</span>{display_file_name}</div></div>
                </div>
                <div class="preview">{image_html}</div>
                <div>
                    <div class="title" style="margin-bottom:10px;">Diagnosis Result</div>
                    <div class="diag {diag_class}"><div class="diag-icon">!</div><div><strong>{verdict}</strong><span>{label}</span></div></div>
                    <div class="conf">
                        <div><div class="label">Confidence Score</div><div class="circle" style="--score:{score * 3.6}deg;"><div class="inner">{score:.1f}%</div></div></div>
                        <div class="ai"><div class="label">AI Interpretation</div>{interpretation}</div>
                    </div>
                    <div class="rec"><strong>Recommendation</strong><br>{recommendation}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_diagnosis_list(image_results):
    # تم إيقاف تفعيل القائمة الرأسية لتوحيد العرض مع المظهر الأفقي للـ Matrix المعروض في image_fb87d7.png
    pass


def render_video_frames(pipeline):
    # تستخدم لعرض شبكة الفريمات المستخلصة من الفيديو أو الصور المرفوعة
    frames = pipeline["extracted_frames"]
    count = min(len(frames), EXTRACTED_FRAMES_DISPLAY_COUNT)
    st.markdown('<div class="panel"><div class="title"><i>▧</i>Most Informative Frames</div>', unsafe_allow_html=True)

    headers = st.columns(count + 1)
    headers[0].markdown("")
    for i in range(count):
        # تفعيل أزرار التنقل التفاعلية العلوية لكل فريم لتماثل الصورة تماماً عند الضغط عليها
        if headers[i + 1].button(f"Frame {i + 1}", key=f"frame_nav_btn_{i}", use_container_width=True):
            st.session_state["selected_image_index"] = i
            st.rerun()

    for title, getter in [
        ("Original<br>Frame", lambda f: cv.cvtColor(f.image_bgr, cv.COLOR_BGR2RGB)),
        ("ROI", lambda f: f.roi_rgb if f.roi_rgb is not None else cv.cvtColor(f.image_bgr, cv.COLOR_BGR2RGB)),
        ("GradCAM<br>(Heatmap)", lambda f: f.heatmap_rgb),
    ]:
        row = st.columns(count + 1)
        row[0].markdown(f"<div class='meta' style='height:75px; display:flex; align-items:center;'>{title}</div>", unsafe_allow_html=True)
        for i in range(count):
            img = getter(frames[i])
            if img is not None:
                row[i + 1].image(img, use_container_width=True)
            else:
                row[i + 1].info("No heatmap")

    score_row = st.columns(count + 1)
    score_row[0].markdown("")
    for i in range(count):
        score_row[i + 1].markdown(f"<div class='meta'>Score: {frames[i].probability:.2f}</div>", unsafe_allow_html=True)

    time_row = st.columns(count + 1)
    time_row[0].markdown("<div class='meta'>Timestamp (sec)</div>", unsafe_allow_html=True)
    for i in range(count):
        time_row[i + 1].markdown(f"<div class='meta'>{frames[i].timestamp_sec:0>7.2f}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_video_report_summary():
    final_class, confidence, _, total_frames, positive_frames = current_result_summary()
    if final_class is None:
        return

    report_path = Path(st.session_state["report_path"])
    diagnosis = "Positive" if is_positive(final_class) else "Negative"
    label = CLASS_LABELS[final_class]
    media_type = "Image" if st.session_state.get("image_results") else "Video"

    st.markdown('<div class="panel"><div class="title"><i>▤</i>Report Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="summary">
            <div class="sumrow"><span>Diagnosis</span><strong>{diagnosis} - {label}</strong></div>
            <div class="sumrow"><span>Confidence</span><strong>{confidence:.1f}%</strong></div>
            <div class="sumrow"><span>Model</span><strong>ResNet18</strong></div>
            <div class="sumrow"><span>Media Type</span><strong>{media_type}</strong></div>
            <div class="sumrow"><span>Positive Frames</span><strong>{positive_frames}</strong></div>
            <div class="sumrow"><span>Top Frames Used</span><strong>{total_frames}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report_path.exists():
        with open(report_path, "rb") as pdf_file:
            b64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64_pdf}" download="{report_path.name}" class="download">Download PDF Report</a>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="ghost">Print Report</div>', unsafe_allow_html=True)
    with c2:
        if st.button("New Examination", use_container_width=True):
            reset_examination()

    st.markdown("</div>", unsafe_allow_html=True)


def render_results():
    image_results = st.session_state.get("image_results", [])

    # عند وجود صور مبرمجة، يتم تجميعها لتعمل داخل واجهة الجريد الموحدة تماماً كالفيديو
    if image_results:
        extracted_frames = [item["result"] for item in image_results]
        pipeline_mock = {"extracted_frames": extracted_frames}
        
        render_exam_overview()
        left, right = st.columns([0.68, 0.32], gap="large")
        with left:
            render_video_frames(pipeline_mock)
        with right:
            render_video_report_summary()
        return

    render_exam_overview()
    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        if st.session_state["video_result"] is not None:
            render_video_frames(st.session_state["video_result"])

    with right:
        render_video_report_summary()


def main():
    init_session_state()
    render_sidebar()
    render_header()
    render_stats()

    if not render_patient_form():
        return

    has_result = (
        bool(st.session_state.get("image_results"))
        or st.session_state["video_result"] is not None
    )

    if has_result:
        render_results()
    else:
        render_upload_area()
        render_exam_overview()


if __name__ == "__main__":
    main()