# Attendance AI - System Architecture & Documentation

Welcome to the **Attendance AI** documentation! This guide is designed to help new developers quickly grasp the project's structure, Machine Learning (ML) pipeline, dataflow, and underlying database architecture. 

---

## 1. Project Overview & Architecture

Attendance AI is a production-ready facial recognition attendance management system built with **Streamlit**. It allows administrators to register students with a single clear facial photo, and subsequently mark attendance by uploading raw video footage (e.g., from a classroom or security camera).

The project strictly follows a modular architecture separating UI, database operations, and computer vision logic:

*   **`app.py`**: The main Streamlit web application. Contains the UI layout, routing across tabs (Registration, Attendance Processing, Dashboard, Manage Students), session state management, and the high-throughput video processing loop.
*   **`aligner.py`**: The Computer Vision (CV) module responsible for facial landmark extraction and affine warp alignment. Ensures faces are geometrically normalized before generating embeddings.
*   **`database.py`**: The Data Persistence layer. Manages all interactions with the Supabase backend. Includes connection resilience (auto-retries for stale connections) and executes the server-side vector matching RPC.
*   **`requirements.txt`**: Project dependencies including `streamlit`, `insightface`, `onnxruntime`, `opencv-python-headless`, `supabase`, and `pandas`.

---

## 2. Machine Learning Pipeline

The system relies on a state-of-the.art facial recognition pipeline to ensure high accuracy and stability. The entire ML process happens within `app.py` and `aligner.py`, utilizing the **InsightFace (`buffalo_l`)** model.

### Step-by-Step Face Processing:
1.  **Face Detection**: The InsightFace model scans an image or video frame and detects human faces, returning bounding boxes.
2.  **Gatekeeping & Quality Checks** (Video processing only):
    *   **Dimension Filter**: Rejects small, background faces (Bounding box must be $\ge 60 \times 60$ pixels).
    *   **Boundary Check**: Rejects chopped faces touching the edges of the frame (10px margin).
    *   **Sharpness (Blur) Check**: Analyzes the aligned face crop using Laplacian variance. Rejects blurry frames (Variance $< 100.0$).
    *   **Spatial Consistency (IoU)**: Computes Intersection over Union (IoU) with faces in the previous processed frame. If the face doesn't spatially align with a previous track ($< 0.3$ IoU), it skips DB lookup to save API calls and avoid transient false positives.
3.  **Landmark Extraction**: Extracts 5 reference points (Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner).
4.  **Affine Warp Alignment**: Uses the 5 landmarks to compute a similarity transformation matrix (`cv2.estimateAffinePartial2D`). The face is then warped (`cv2.warpAffine`) to a deterministic $112 \times 112$ pixel canonical crop. *This removes geometric variance (tilt, scale) and guarantees embedding stability.*
5.  **Embedding Extraction**: The aligned face is passed through the ArcFace network to generate a **512-dimensional vector embedding** (normalized).
6.  **Matching**: The 512D embedding is sent to the Supabase database where the cosine similarity calculation happens entirely on the server.

---

## 3. Database Schema & Vector Search

The backend is powered by **Supabase (PostgreSQL)** leveraging the `pgvector` extension for lightning-fast server-side vector math.

### Tables:
*   **`students`**: Stores registered users.
    *   `student_id` (PK), `name`, `roll_number`, `department`, `created_at`.
    *   `embedding`: A `vector(512)` field holding the registered face embedding.
*   **`lecture_sessions`**: Tracks active and past lectures.
    *   `id` (PK), `subject`, `status` (active/ended), `started_at`, `ended_at`.
*   **`attendance_logs`**: Joins students to sessions.
    *   `id` (PK), `student_id` (FK), `session_id` (FK), `status` (Present/Partial/Absent), `marked_at`.

### Vector Matching (The `match_faces` RPC)
Instead of fetching all vectors to the Python backend, matching is done inside Postgres via a Remote Procedure Call (RPC). It uses **Cosine Distance** (`<=>`):
```sql
Similarity = 1 - (students.embedding <=> query_embedding)
```
The RPC `match_faces(query_embedding, match_threshold, match_count)` returns the closest students whose similarity exceeds the `match_threshold`.

---

## 4. Application Workflows & Dataflow

### A. Student Registration
1.  Admin inputs student details and uploads/captures a photo.
2.  Pipeline detects face $\rightarrow$ aligns face $\rightarrow$ generates 512D embedding.
3.  Embedding and details are inserted into the `students` table.

### B. High-Throughput Video Processing
1.  Admin starts a "Lecture Session" (creates a `lecture_sessions` record).
2.  Admin uploads a video and sets *Confidence Threshold* and *Frame Sampling Rate*.
3.  The app writes the video to a temporary file and opens it with OpenCV.
4.  **Loop**: For every $N$th frame:
    *   Detect faces, run quality gatekeepers.
    *   Extract embeddings for valid faces.
    *   Call `match_faces` RPC on Supabase.
    *   Log matches in a session cache (`student_window_matches`) mapped to 5-second temporal windows.
    *   Track "Unknown" faces (comparing new unknown embeddings with previously seen unknown embeddings using cosine similarity $> 0.70$).
5.  **Finalization**: Calculates presence ratios. 
    *   Inserts `attendance_logs` into Supabase for each student.
    *   Generates a final CSV report and UI dataframe.

### C. Unknown Face Resolution (Conflict Resolver)
If unregistered faces are detected during a session, they are saved in `st.session_state.unresolved_faces`. The admin can manually assign these faces to registered students.
*   **Action**: Resolving a face immediately updates the student's primary embedding in the database (improving future recognition) and retroactively updates their attendance log for that session to "Present".

---

## 5. Key Parameters & Criteria Cheat Sheet

Here are the critical thresholds hardcoded or configurable in the system that govern the ML accuracy and business logic:

| Parameter | Value | Location | Description |
| :--- | :--- | :--- | :--- |
| **Aligned Face Size** | `112x112` | `aligner.py` | Canonical crop size required by ArcFace model. |
| **Min Bounding Box** | `60x60 px` | `app.py` (Video loop) | Ignores faces smaller than this to prevent noisy low-res matches. |
| **Boundary Margin** | `10 px` | `app.py` (Video loop) | Discards faces within 10 pixels of the video frame edge. |
| **Blur Variance Threshold** | `100.0` | `app.py` (Video loop) | Minimum Laplacian variance. Lower values mean the face is too blurry. |
| **IoU Spatial Threshold** | `0.3` | `app.py` (Video loop) | Minimum bounding box overlap from the previous frame to trigger a DB match lookup. |
| **Match Confidence** | `0.65` | `app.py` (Slider) | Default Cosine similarity threshold for a positive DB match. |
| **Frame Sampling Rate** | `5` | `app.py` (Slider) | Processes 1 frame every 5 frames to speed up video processing. |
| **Attendance Window** | `5.0 sec` | `app.py` (Finalization) | Video is bucketed into 5-second windows to calculate presence duration. |
| **"Present" Criteria** | $\ge 60\%$ | `app.py` (Finalization) | Student must be seen in at least 60% of the total lecture windows. |
| **"Partial" Criteria** | $\ge 15\%$ | `app.py` (Finalization) | Student was seen between 15% and 59% of the lecture. |
| **Unknown Merge Threshold** | `0.70` | `app.py` (Video loop) | Similarity threshold to group multiple unknown face detections into a single "Unknown Person". |

---
*Created by the AI Assistant to provide a comprehensive structural overview of the Attendance AI platform.*
