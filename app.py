"""
app.py — Attendance-AI: Main Streamlit Application
====================================================
A production-ready attendance management system using:
  - InsightFace (buffalo_l) for face detection & ArcFace embeddings
  - Affine warp alignment for deterministic 112×112 face crops
  - Supabase (pgvector) for server-side 512D cosine similarity matching

Tabs:
  1. Registration — Onboard students with face photos
  2. Attendance  — Process video files for bulk attendance marking
  3. Dashboard   — View registered students and today's attendance
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from PIL import Image
from io import BytesIO

from aligner import align_face, extract_landmarks_from_face
from database import (
    register_student,
    match_face,
    log_attendance,
    update_attendance_status,
    update_student_embedding,
    get_all_students,
    update_student,
    delete_student,
    start_session,
    end_session,
    get_all_sessions,
    get_session_report,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Attendance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# ══════════════════════════════════════════════════════════════
# MODEL INITIALIZATION (Singleton via session_state)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Loading InsightFace model (buffalo_l)...")
def load_face_model():
    """Load the InsightFace FaceAnalysis model exactly once (CPU)."""
    try:
        # pyrefly: ignore [missing-import]
        import insightface
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        model.prepare(ctx_id=0, det_size=(640, 640))
        return model
    except Exception as e:
        st.error(
            f"❌ **Failed to load InsightFace model.**\n\n"
            f"Make sure `insightface` and `onnxruntime` are installed.\n\n"
            f"Error: `{e}`"
        )
        st.stop()


face_model = load_face_model()


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🎓 Attendance AI")
st.markdown("Face-recognition powered attendance system")
st.markdown("---")


# ══════════════════════════════════════════════════════════════
# HELPER: Compute IoU
# ══════════════════════════════════════════════════════════════
def compute_iou(box1, box2):
    """Compute Intersection over Union for two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)


# ══════════════════════════════════════════════════════════════
# HELPER: Detect faces and extract embeddings
# ══════════════════════════════════════════════════════════════
def process_face_from_image(image_bgr: np.ndarray):
    """
    Detect a face, align it, and return (aligned_crop, embedding).
    Returns (None, None, error_msg) on failure.
    """
    faces = face_model.get(image_bgr)
    if not faces:
        return None, None, "No face detected in the image. Ensure the face is clearly visible and well-lit."

    if len(faces) > 1:
        # Use the largest face (by bounding box area)
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

    face = faces[0]

    # Extract 5-point landmarks
    landmarks = extract_landmarks_from_face(face)
    if landmarks is None:
        return None, None, "Could not extract facial landmarks. Try a clearer photo with the face looking forward."

    # Affine warp alignment → 112×112 canonical crop
    aligned = align_face(image_bgr, landmarks)
    if aligned is None:
        return None, None, "Face alignment failed. The landmarks may be degenerate — try a different photo."

    # Get the 512D ArcFace embedding
    embedding = face.normed_embedding
    if embedding is None or embedding.shape != (512,):
        return None, None, f"Embedding extraction failed. Got shape: {embedding.shape if embedding is not None else 'None'}"

    return aligned, embedding.astype(np.float32), None


# ══════════════════════════════════════════════════════════════
# SIDEBAR / NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎓 :violet[Attendance AI]")
    
    selected_page = st.radio(
        "Navigation",
        ["Registration", "Attendance Processing", "Dashboard", "Manage Students"]
    )
    
    st.divider()
    st.markdown("🟢 :green[**System Online**]")
    st.markdown("🟣 :violet[**Recognition Ready**]")
    
    st.divider()
    st.markdown("👤 **Admin User**")
    st.caption("Workspace Admin")
    
    st.markdown("---")
    st.markdown("### 🛠️ Admin Controls")
    if st.button("🔄 Reset App Session", use_container_width=True):
        st.session_state.clear()
        st.toast("Session state cleared!", icon="🔄")
        
    if st.button("🧹 Clear Global Cache", use_container_width=True):
        st.cache_resource.clear()
        st.toast("Global cache cleared!", icon="🧹")

# ──────────────────────────────────────────────────────────────
# MAIN PAGES
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# PAGE 1: REGISTRATION / ONBOARDING
# ──────────────────────────────────────────────────────────────
if selected_page == "Registration":
    st.header("Register Student")
    st.write("Add a new student to the facial recognition database.")

    col_form, col_photo = st.columns([5, 4], gap="large")

    with col_form:
        st.subheader("Student Information")
        student_id = st.text_input("Student ID", placeholder="STU-001")
        student_name = st.text_input("Full Name", placeholder="John Doe")
        roll_number = st.text_input("Roll Number", placeholder="CS2023-001")
        department = st.text_input("Department", placeholder="Computer Science")

        st.write("")
        upload_method = st.radio(
            "Image Source", ["Upload File", "Camera"],
            horizontal=True, label_visibility="collapsed"
        )
        st.write("")
        
    with col_photo:
        st.subheader("Face Photo")
        
        image_file = None
        if upload_method == "Upload File":
            image_file = st.file_uploader(
                "Drop image here (JPG, PNG)",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed"
            )
        else:
            image_file = st.camera_input("Capture Face Photo", label_visibility="collapsed")
            
        st.divider()
        st.subheader("Preview")
        if not image_file:
            st.info("Preview will appear here")
        else:
            pil_img = Image.open(image_file).convert("RGB")
            st.image(pil_img, use_container_width=True)
            
        st.divider()
        st.subheader("Guidelines")
        st.markdown("""
        - Clear, front-facing photo
        - Good lighting on the face
        - No glasses or masks
        - Neutral expression
        """)

    st.divider()
    
    register_btn = st.button(
        "Register Student", type="primary",
        use_container_width=True, disabled=not (student_id and student_name and image_file and roll_number and department)
    )
    
    if register_btn and image_file:
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        with st.status("Registering Student...", expanded=True) as status:
            st.write("🔍 Detecting face & generating embedding...")
            aligned, embedding, error = process_face_from_image(img_bgr)

            if error:
                status.update(label="Detection Failed", state="error", expanded=True)
                st.error(f"❌ **Detection Failed:** {error}")
            else:
                # Show the aligned 112×112 crop
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                st.image(aligned_rgb, caption="Aligned 112×112 Crop", width=150)

                # Persist to Supabase
                try:
                    st.write("💾 Saving to database...")
                    result = register_student(student_id, student_name, embedding, roll_number, department)
                    status.update(label="Student Registered Successfully!", state="complete", expanded=False)
                    st.toast(f"Welcome, {student_name}! Registered successfully.", icon="✅")
                    st.balloons()
                except ValueError as e:
                    status.update(label="Validation Error", state="error", expanded=True)
                    st.error(f"⚠️ **Validation Error:** {e}")
                except Exception as e:
                    status.update(label="Database Error", state="error", expanded=True)
                    st.error(f"❌ **Database Error:** {e}")


# ──────────────────────────────────────────────────────────────
# PAGE 2: LECTURE SESSIONS & ATTENDANCE
# ──────────────────────────────────────────────────────────────
elif selected_page == "Attendance Processing":
    st.header("Lecture Sessions")
    st.write("Process attendance video and resolve conflicts.")
    
    if "active_session" not in st.session_state:
        st.session_state.active_session = None

    if st.session_state.active_session is None:
        st.info("Start a new lecture session to process attendance video.")
        with st.form("start_session_form"):
            subject_name = st.text_input("Class/Subject Name", placeholder="e.g. Computer Science 101 - Lecture 5")
            if st.form_submit_button("Start Lecture Session", type="primary") and subject_name:
                try:
                    new_session = start_session(subject_name)
                    st.session_state.active_session = new_session
                    st.success(f"Started session: {subject_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start session: {e}")
    else:
        active_sess = st.session_state.active_session
        st.subheader(f"🟢 Active Session: {active_sess['subject']}")
        
        if st.button("🔴 End and Save Session", type="primary"):
            try:
                end_session(active_sess["id"])
                st.session_state.active_session = None
                st.success("Session ended successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to end session: {e}")
                
        st.divider()
        st.subheader("Process Attendance Video")
        col_config, col_status = st.columns([1, 1], gap="large")

        with col_config:
            video_file = st.file_uploader(
                "Upload Video File",
                type=["mp4", "avi", "mov"],
                help="Supported formats: MP4, AVI, MOV. Max 500MB."
            )
    
            threshold = st.slider(
                "Matching Confidence Threshold",
                min_value=0.3, max_value=0.95, value=0.65, step=0.05,
                help="Higher = stricter matching (fewer false positives). "
                     "0.65 is recommended for ArcFace."
            )
    
            frame_step = st.slider(
                "Frame Sampling Rate",
                min_value=1, max_value=30, value=5,
                help="Process every Nth frame. Higher = faster but may miss faces."
            )
    
            process_btn = st.button(
                "▶️ Start Processing", type="primary",
                use_container_width=True, disabled=not video_file
            )
    
        with col_status:
            if process_btn and video_file:
                # Write uploaded video to a temp file for OpenCV
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                tfile.flush()
                temp_path = tfile.name
                tfile.close()
    
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    st.error("❌ **Failed to open video file.** The file may be corrupted or in an unsupported codec.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
                    st.info(f"📹 **Video Info:** {total_frames} frames @ {fps:.1f} FPS ({total_frames/fps:.1f}s) | Processing every **{frame_step}** frame(s) → ~{total_frames // frame_step} frames to analyze")
    
                    progress_bar = st.progress(0, text="Initializing...")
                    status_text = st.empty()
                    debug_expander = st.expander("🔧 Debug Log (click to expand)", expanded=False)
                    debug_logs = []
    
                    # Session cache: prevent duplicate DB hits per video run
                    seen_ids: set = set()
                    matched_students: list = []
                    frames_processed = 0
                    faces_detected = 0
                    unknown_faces_count = 0
                    session_unknown_embeddings = []
                    session_unknown_crops = []
                    student_window_matches = {}
                    student_names = {}
                    prev_frame_bboxes = []
    
                    frame_indices = list(range(0, total_frames, frame_step))
    
                    for i, frame_idx in enumerate(frame_indices):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret:
                            continue
    
                        frames_processed += 1
                        pct = (i + 1) / len(frame_indices)
                        progress_bar.progress(pct, text=f"Frame {frame_idx}/{total_frames} ({pct*100:.0f}%)")
                        
                        current_frame_bboxes = []

                        # Detect all faces in this frame
                        try:
                            faces = face_model.get(frame)
                        except Exception as e:
                            status_text.warning(f"⚠️ Frame {frame_idx}: detection error — {e}")
                            continue
    
                        if not faces:
                            continue
    
                        faces_detected += len(faces)
    
                        for face in faces:
                            # 1. Dimension Filter: skip small background faces
                            bbox = face.bbox
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            if width < 60 or height < 60:
                                continue

                            landmarks = extract_landmarks_from_face(face)
                            if landmarks is None:
                                if len(debug_logs) < 10:
                                    debug_logs.append(f"Frame {frame_idx}: landmarks=None, skipping")
                                continue
                                
                            # Gatekeeper 1: Boundary Check (10px margin)
                            height, width = frame.shape[:2]
                            margin = 10
                            is_chopped = False
                            for pt in landmarks:
                                x, y = pt
                                if x < margin or x > width - margin or y < margin or y > height - margin:
                                    is_chopped = True
                                    break
                            if is_chopped:
                                if len(debug_logs) < 10:
                                    debug_logs.append(f"Frame {frame_idx}: face chopped at boundary, skipping")
                                continue
    
                            aligned = align_face(frame, landmarks)
                            if aligned is None:
                                if len(debug_logs) < 10:
                                    debug_logs.append(f"Frame {frame_idx}: alignment failed, skipping")
                                continue
                                
                            # Gatekeeper 2: Sharpness Check
                            blur_val = cv2.Laplacian(aligned, cv2.CV_64F).var()
                            if blur_val < 100.0:
                                if len(debug_logs) < 10:
                                    debug_logs.append(f"Frame {frame_idx}: blur variance {blur_val:.2f} < 100.0, skipping")
                                continue
                                
                            # If it passed quality gates, we record it for future IoU tracking
                            current_frame_bboxes.append(bbox)
                            
                            # Gatekeeper 3: Spatial Consistency (IoU)
                            max_iou = 0.0
                            for prev_box in prev_frame_bboxes:
                                iou = compute_iou(bbox, prev_box)
                                if iou > max_iou:
                                    max_iou = iou
                            
                            if max_iou < 0.3:
                                if len(debug_logs) < 10:
                                    debug_logs.append(f"Frame {frame_idx}: IoU {max_iou:.2f} < 0.3, skipping lookup")
                                continue
    
                            emb = face.normed_embedding
                            if emb is None or emb.shape != (512,):
                                if len(debug_logs) < 10:
                                    debug_logs.append(f"Frame {frame_idx}: embedding={emb.shape if emb is not None else None}")
                                continue
    
                            emb = emb.astype(np.float32)
    
                            # Debug: log embedding stats for first few faces
                            if len(debug_logs) < 10:
                                debug_logs.append(
                                    f"Frame {frame_idx}: emb norm={np.linalg.norm(emb):.4f}, "
                                    f"min={emb.min():.4f}, max={emb.max():.4f}, "
                                    f"dtype={emb.dtype}"
                                )
    
                            # Query Supabase for match
                            try:
                                matches = match_face(emb, threshold=threshold, max_matches=3)
                                if len(debug_logs) < 15:
                                    debug_logs.append(
                                        f"  → RPC returned {len(matches)} match(es): "
                                        f"{matches if matches else 'empty'}"
                                    )
                            except Exception as e:
                                if len(debug_logs) < 15:
                                    debug_logs.append(f"  → RPC ERROR: {e}")
                                status_text.warning(f"⚠️ Match RPC error at frame {frame_idx}: {e}")
                                continue
    
                            if matches:
                                match = matches[0]
                                sid = match["student_id"]
                                seen_ids.add(sid)
                                
                                timestamp_sec = frame_idx / fps
                                window_idx = int(timestamp_sec // 5.0)
                                if sid not in student_window_matches:
                                    student_window_matches[sid] = set()
                                student_window_matches[sid].add(window_idx)
                                
                                student_names[sid] = match["name"]
                            else:
                                is_new_unknown = True
                                for idx, unknown_emb in enumerate(session_unknown_embeddings):
                                    sim = np.dot(emb, unknown_emb) / (np.linalg.norm(emb) * np.linalg.norm(unknown_emb))
                                    if sim > 0.70:
                                        is_new_unknown = False
                                        break
                                
                                if is_new_unknown:
                                    session_unknown_embeddings.append(emb)
                                    session_unknown_crops.append(aligned)
                                    unknown_faces_count = len(session_unknown_embeddings)
                                    matched_students.append({
                                        "Student ID": f"UNKNOWN-{unknown_faces_count}",
                                        "Name": "Unregistered Face",
                                        "Similarity": "N/A",
                                        "Frame": frame_idx,
                                    })
                                    
                        prev_frame_bboxes = current_frame_bboxes
    
                        # Live update matched count
                        status_text.markdown(
                            f"🔍 Processed **{frames_processed}** frames | "
                            f"👤 Detected **{faces_detected}** faces | "
                            f"✅ Matched **{len(seen_ids)}** students | "
                            f"❓ Unknown **{unknown_faces_count}**"
                        )
    
                    cap.release()
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
    
                    # Compute Final Attendance Statuses
                    status_text.markdown("🔄 Finalizing attendance logs and generating report...")
                    all_registered_students = get_all_students()
                    
                    import math
                    total_duration = total_frames / fps if fps > 0 else 0
                    total_windows = max(1, math.ceil(total_duration / 5.0))
                    
                    final_report = []
                    for s in all_registered_students:
                        sid = s["student_id"]
                        matched_windows = len(student_window_matches.get(sid, set()))
                        ratio = matched_windows / total_windows
                        
                        if ratio >= 0.60:
                            status = "Present"
                        elif ratio >= 0.15:
                            status = "Partial"
                        else:
                            status = "Absent"
                            
                        # Log to database
                        try:
                            log_attendance(sid, active_sess["id"], status=status)
                        except Exception as e:
                            st.warning(f"⚠️ Log error for {sid}: {e}")
                            
                        final_report.append({
                            "Student ID": sid,
                            "Name": s["name"],
                            "Presence Ratio": f"{ratio:.1%}",
                            "Status": status,
                        })
                        
                    # Add unknown faces to final report
                    for unk in matched_students:
                        final_report.append({
                            "Student ID": unk["Student ID"],
                            "Name": unk["Name"],
                            "Presence Ratio": "N/A",
                            "Status": "Unknown",
                        })
                        
                    matched_students = final_report
                    
                    # Store unresolved faces in session state
                    if session_unknown_embeddings:
                        st.session_state.unresolved_faces = []
                        for i, (emb, crop) in enumerate(zip(session_unknown_embeddings, session_unknown_crops)):
                            st.session_state.unresolved_faces.append({
                                "id": f"UNKNOWN-{i+1}",
                                "embedding": emb,
                                "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            })
                            
                    progress_bar.progress(1.0, text="✅ Processing complete!")
    
                    # Show debug log
                    with debug_expander:
                        for log_line in debug_logs:
                            st.text(log_line)
    
                    # ── Results ──
                    st.markdown("---")
                    st.markdown("### 📋 Processing Results")
    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Frames Analyzed", frames_processed)
                    c2.metric("Faces Detected", faces_detected)
                    c3.metric("Students Matched", len(seen_ids))
                    c4.metric("Unknown Faces", unknown_faces_count)
    
                    if matched_students:
                        df = pd.DataFrame(matched_students)
                        
                        def highlight_status(val):
                            if val == 'Present':
                                color = 'rgba(34, 197, 94, 0.2)'
                            elif val == 'Partial':
                                color = 'rgba(245, 158, 11, 0.2)'
                            elif val == 'Absent':
                                color = 'rgba(239, 68, 68, 0.2)'
                            else:
                                color = 'transparent'
                            return f'background-color: {color}'
                            
                        styled_df = df.style.map(highlight_status, subset=['Status'])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Attendance CSV",
                            data=csv,
                            file_name="attendance_report.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.warning("⚠️ **No matches found.** Possible reasons:\n- No students registered yet\n- Confidence threshold too high (try lowering to 0.5)\n- Faces in video not clear enough for matching")
    
            elif not video_file:
                st.info("📹 **Upload a video file** to begin attendance processing. The system will detect faces frame-by-frame and match them against registered students using cosine similarity.")

        if "unresolved_faces" in st.session_state and st.session_state.unresolved_faces:
            st.markdown("---")
            st.markdown("### ❓ Unresolved Faces (Conflict Resolver)")
            st.markdown("Assign a registered student to these unknown faces to update their attendance and improve the model.")
            
            try:
                all_students = get_all_students()
                student_opts = {s["student_id"]: f"{s['student_id']} - {s['name']}" for s in all_students}
                
                for face_data in st.session_state.unresolved_faces:
                    c_img, c_sel, c_btn = st.columns([1, 2, 1])
                    with c_img:
                        st.image(face_data["crop"], width=112, caption=face_data["id"])
                    with c_sel:
                        selected_sid = st.selectbox(
                            f"Select identity for {face_data['id']}",
                            options=[""] + list(student_opts.keys()),
                            format_func=lambda x: student_opts[x] if x else "Select a student...",
                            key=f"sel_{face_data['id']}"
                        )
                    with c_btn:
                        st.write("") # spacer
                        st.write("")
                        if st.button("Override & Resolve", key=f"btn_{face_data['id']}", type="primary", disabled=not selected_sid):
                            if selected_sid:
                                try:
                                    # 1. Update embedding in DB
                                    update_student_embedding(selected_sid, face_data["embedding"])
                                    # 2. Update attendance status to Present
                                    if st.session_state.active_session:
                                        update_attendance_status(selected_sid, st.session_state.active_session["id"], "Present")
                                    
                                    st.success(f"Resolved {face_data['id']} as {student_opts[selected_sid]}!")
                                    
                                    # Remove from session state
                                    st.session_state.unresolved_faces = [f for f in st.session_state.unresolved_faces if f["id"] != face_data["id"]]
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to resolve: {e}")
            except Exception as e:
                st.error(f"Could not load students for resolver: {e}")


# ──────────────────────────────────────────────────────────────
# PAGE 3: DASHBOARD
# ──────────────────────────────────────────────────────────────
elif selected_page == "Dashboard":
    st.header("Dashboard & Reports")

    col_students, col_attendance = st.columns(2, gap="large")

    with col_students:
        st.subheader("Registered Students")
        try:
            students = get_all_students()
            if students:
                st.metric("Total Registered", len(students))
                st.dataframe(
                    pd.DataFrame(students),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No students registered yet. Go to the Registration tab to add students.")
        except Exception as e:
            st.error(f"❌ Failed to load students: {e}")

    with col_attendance:
        st.subheader("Lecture History")
        if st.button("🔄 Refresh", key="refresh_attendance"):
            st.rerun()
        try:
            sessions = get_all_sessions()
            if sessions:
                session_options = {s['id']: f"{s['subject']} ({s['started_at'][:10]})" for s in sessions}
                selected_sess_id = st.selectbox("Select a Session", options=list(session_options.keys()), format_func=lambda x: session_options[x])
                
                if selected_sess_id:
                    report = get_session_report(selected_sess_id)
                    if report:
                        df = pd.DataFrame(report)
                        
                        # Show stats
                        present_count = sum(1 for r in report if r["Status"] == "Present")
                        st.markdown(f"**Present:** {present_count} / {len(report)}")
                        
                        # Show dataframe with color coding for Status
                        def highlight_status(val):
                            if val == 'Present':
                                color = 'rgba(34, 197, 94, 0.2)'
                            elif val == 'Partial':
                                color = 'rgba(245, 158, 11, 0.2)'
                            else:
                                color = 'rgba(239, 68, 68, 0.2)'
                            return f'background-color: {color}'
                            
                        styled_df = df.style.map(highlight_status, subset=['Status'])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)

                        # Provide export
                        csv = df.to_csv(index=False).encode("utf-8")
                        sess_name = session_options[selected_sess_id].replace(" ", "_").replace("/", "-")
                        st.download_button(
                            "📥 Download Attendance CSV",
                            data=csv,
                            file_name=f"attendance_{sess_name}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.info("No report available.")
            else:
                st.info("No lecture sessions found. Start a session in the Attendance tab.")
        except Exception as e:
            st.error(f"❌ Failed to load sessions: {e}")

# ──────────────────────────────────────────────────────────────
# PAGE 4: STUDENT MANAGEMENT
# ──────────────────────────────────────────────────────────────
elif selected_page == "Manage Students":
    st.header("Manage Students")
    st.write("Edit profile details or delete students from the system.")
    
    try:
        students = get_all_students()
        if not students:
            st.info("No students registered yet.")
        else:
            student_options = {s['student_id']: f"{s['student_id']} - {s['name']}" for s in students}
            selected_id = st.selectbox("Select Student to Manage", options=list(student_options.keys()), format_func=lambda x: student_options[x])
            
            if selected_id:
                selected_student = next(s for s in students if s['student_id'] == selected_id)
                
                st.markdown("---")
                col_edit, col_delete = st.columns(2, gap="large")
                
                with col_edit:
                    st.subheader("Edit Profile")
                    with st.form("edit_student_form"):
                        new_name = st.text_input("Name", value=selected_student.get("name", ""))
                        new_roll = st.text_input("Roll Number", value=selected_student.get("roll_number", ""))
                        new_dept = st.text_input("Department", value=selected_student.get("department", ""))
                        
                        update_submit = st.form_submit_button("💾 Save Changes", type="primary")
                        
                        if update_submit:
                            updates = {}
                            if new_name != selected_student.get("name"): updates["name"] = new_name
                            if new_roll != selected_student.get("roll_number"): updates["roll_number"] = new_roll
                            if new_dept != selected_student.get("department"): updates["department"] = new_dept
                            
                            if updates:
                                try:
                                    update_student(selected_id, updates)
                                    st.success("✅ Student updated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Update failed: {e}")
                            else:
                                st.info("No changes made.")
                                
                    st.divider()
                    with st.expander("📸 Update Face Photo"):
                        st.write("Upload a new photo to replace the existing facial recognition data for this student.")
                        upd_method = st.radio("Image Source", ["Upload File", "Camera"], horizontal=True, key=f"src_{selected_id}")
                        
                        new_img_file = None
                        if upd_method == "Upload File":
                            new_img_file = st.file_uploader("Drop new image", type=["jpg", "jpeg", "png", "webp"], key=f"up_{selected_id}")
                        else:
                            new_img_file = st.camera_input("Capture New Face", key=f"cam_{selected_id}")
                            
                        if new_img_file:
                            pil_upd = Image.open(new_img_file).convert("RGB")
                            st.image(pil_upd, width=150, caption="Preview")
                            if st.button("Generate & Save New Embedding", type="primary", key=f"btn_upd_{selected_id}"):
                                img_bgr = cv2.cvtColor(np.array(pil_upd), cv2.COLOR_RGB2BGR)
                                with st.spinner("Processing new face..."):
                                    aligned, embedding, error = process_face_from_image(img_bgr)
                                if error:
                                    st.error(f"❌ **Detection Failed:** {error}")
                                else:
                                    try:
                                        update_student_embedding(selected_id, embedding)
                                        st.success("✅ Face data updated successfully in Supabase!")
                                    except Exception as e:
                                        st.error(f"❌ Database error: {e}")
                
                with col_delete:
                    st.markdown("#### Danger Zone")
                    st.warning("Deleting a student will also delete all their attendance records permanently.")
                    if st.button("🗑️ Delete Student", type="primary", use_container_width=True):
                        try:
                            delete_student(selected_id)
                            st.success(f"✅ Student {selected_id} deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Deletion failed: {e}")

    except Exception as e:
        st.error(f"❌ Failed to load students: {e}")

