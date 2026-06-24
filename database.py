"""
database.py — Supabase Data Persistence Layer
===============================================
Handles all Supabase PostgreSQL interactions:
  - Student registration with 512D vector embedding
  - Server-side face matching via match_faces() RPC
  - Attendance log insertion and retrieval

Includes automatic retry logic for stale connection resets (WinError 10054).
"""

import streamlit as st
from supabase import create_client, Client
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, TypeVar
import numpy as np
import time
import functools

T = TypeVar("T")

# ── Connection-resilient client management ─────────────────────
_client_instance: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create the Supabase client (recreatable on connection reset)."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except KeyError as e:
        raise KeyError(
            f"Missing Supabase credential in .streamlit/secrets.toml: {e}."
        ) from e
    try:
        _client_instance = create_client(url, key)
        return _client_instance
    except Exception as e:
        raise ConnectionError(f"Failed to create Supabase client: {e}") from e


def _reset_client():
    """Force-recreate the client on next call (clears stale connection)."""
    global _client_instance
    _client_instance = None


def _with_retry(operation: Callable[[], T], max_retries: int = 3) -> T:
    """
    Execute a Supabase operation with automatic retry on connection reset.
    On WinError 10054 / ConnectionError, resets the client and retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as e:
            err_str = str(e)
            is_conn_error = any(k in err_str for k in [
                "10054", "forcibly closed", "ConnectionReset",
                "RemoteDisconnected", "ConnectionError",
            ])
            if is_conn_error and attempt < max_retries:
                _reset_client()
                time.sleep(1 * attempt)  # Backoff: 1s, 2s
                continue
            raise


def register_student(
    student_id: str,
    name: str,
    embedding: np.ndarray,
    roll_number: str = "",
    department: str = "",
) -> Dict:
    """Insert a new student with their 512D face embedding and profile info."""
    if embedding.shape != (512,):
        raise ValueError(f"Need 512D vector, got {embedding.shape}.")
    embedding_list = embedding.astype(np.float32).tolist()

    row = {
        "student_id": student_id,
        "name": name,
        "embedding": embedding_list,
    }
    if roll_number:
        row["roll_number"] = roll_number
    if department:
        row["department"] = department

    def _do_insert():
        client = get_supabase_client()
        result = client.table("students").insert(row).execute()
        if not result.data:
            raise RuntimeError(f"Insert returned no data for '{student_id}'.")
        return result.data[0]

    try:
        return _with_retry(_do_insert)
    except Exception as e:
        err = str(e).lower()
        if "duplicate" in err or "unique" in err or "23505" in err:
            raise ValueError(f"Student ID '{student_id}' or roll number already exists.") from e
        raise RuntimeError(f"Registration failed for '{name}': {e}") from e


def match_face(embedding: np.ndarray, threshold: float = 0.65, max_matches: int = 1) -> List[Dict]:
    """Query match_faces RPC for cosine similarity matches."""
    if embedding.shape != (512,):
        raise ValueError(f"Need 512D vector, got {embedding.shape}.")
    embedding_list = embedding.astype(np.float32).tolist()

    # pgvector RPC expects the vector as a string "[0.1, 0.2, ...]"
    embedding_str = "[" + ",".join(str(v) for v in embedding_list) + "]"

    def _do_match():
        client = get_supabase_client()
        result = client.rpc("match_faces", {
            "query_embedding": embedding_str,
            "match_threshold": float(threshold),
            "match_count": int(max_matches),
        }).execute()
        return result.data if result.data else []

    try:
        return _with_retry(_do_match)
    except Exception as e:
        raise RuntimeError(f"Face matching RPC failed: {e}") from e


def log_attendance(student_id: str, session_id: str, status: str = "Present") -> Dict:
    """Insert an attendance record for a matched student in a session."""
    def _do_log():
        client = get_supabase_client()
        result = client.table("attendance_logs").insert(
            {"student_id": student_id, "session_id": session_id, "status": status}
        ).execute()
        if not result.data:
            raise RuntimeError(f"No data returned for '{student_id}'.")
        return result.data[0]

    try:
        return _with_retry(_do_log)
    except Exception as e:
        if "foreign key" in str(e).lower() or "23503" in str(e):
            raise ValueError(f"Student '{student_id}' or session '{session_id}' not found.") from e
        raise RuntimeError(f"Attendance log failed: {e}") from e


def update_attendance_status(student_id: str, session_id: str, status: str) -> Dict:
    """Update an existing attendance record's status."""
    def _do_update():
        client = get_supabase_client()
        result = client.table("attendance_logs").update(
            {"status": status}
        ).eq("student_id", student_id).eq("session_id", session_id).execute()
        if not result.data:
            raise RuntimeError(f"No log found for '{student_id}'.")
        return result.data[0]

    try:
        return _with_retry(_do_update)
    except Exception as e:
        raise RuntimeError(f"Attendance status update failed: {e}") from e


def start_session(subject: str) -> Dict:
    """Create a new lecture session."""
    def _do_start():
        client = get_supabase_client()
        result = client.table("lecture_sessions").insert(
            {"subject": subject}
        ).execute()
        if not result.data:
            raise RuntimeError("Failed to start session.")
        return result.data[0]
        
    try:
        return _with_retry(_do_start)
    except Exception as e:
        raise RuntimeError(f"Failed to start session: {e}") from e


def end_session(session_id: str) -> Dict:
    """End an active lecture session."""
    ended_at = datetime.now(timezone.utc).isoformat()
    def _do_end():
        client = get_supabase_client()
        result = client.table("lecture_sessions").update(
            {"status": "ended", "ended_at": ended_at}
        ).eq("id", session_id).execute()
        if not result.data:
            raise RuntimeError(f"Session '{session_id}' not found.")
        return result.data[0]
        
    try:
        return _with_retry(_do_end)
    except Exception as e:
        raise RuntimeError(f"Failed to end session: {e}") from e


def get_all_sessions() -> List[Dict]:
    """Fetch all lecture sessions, ordered by started_at descending."""
    def _do_fetch():
        client = get_supabase_client()
        result = (
            client.table("lecture_sessions")
            .select("*")
            .order("started_at", desc=True)
            .execute()
        )
        return result.data if result.data else []
        
    try:
        return _with_retry(_do_fetch)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch sessions: {e}") from e


def get_session_report(session_id: str) -> List[Dict]:
    """
    Fetch ALL registered students and indicate if they were present or absent
    for the given session_id.
    """
    def _do_fetch():
        client = get_supabase_client()
        # 1. Fetch all students
        students_res = client.table("students").select("student_id, name, roll_number, department").execute()
        students = students_res.data if students_res.data else []
        
        # 2. Fetch attendance logs for this session
        logs_res = client.table("attendance_logs").select("student_id, status").eq("session_id", session_id).execute()
        student_status_map = {log["student_id"]: log["status"] for log in (logs_res.data if logs_res.data else [])}
        
        # 3. Combine into report
        report = []
        for s in students:
            report.append({
                "Student ID": s["student_id"],
                "Roll Number": s.get("roll_number", ""),
                "Name": s["name"],
                "Department": s.get("department", ""),
                "Status": student_status_map.get(s["student_id"], "Absent")
            })
        
        # Sort by Roll Number then Name
        report.sort(key=lambda x: (x["Roll Number"], x["Name"]))
        return report

    try:
        return _with_retry(_do_fetch)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch session report: {e}") from e


def get_all_students() -> List[Dict]:
    """Fetch all registered students (without embeddings)."""
    def _do_fetch():
        client = get_supabase_client()
        result = (
            client.table("students")
            .select("student_id, name, roll_number, department, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data if result.data else []

    try:
        return _with_retry(_do_fetch)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch students: {e}") from e


def update_student(student_id: str, updates: Dict) -> Dict:
    """Update a student's profile fields (name, roll_number, department)."""
    allowed = {"name", "roll_number", "department"}
    filtered = {k: v for k, v in updates.items() if k in allowed and v is not None}
    if not filtered:
        raise ValueError("No valid fields to update.")

    def _do_update():
        client = get_supabase_client()
        result = (
            client.table("students")
            .update(filtered)
            .eq("student_id", student_id)
            .execute()
        )
        if not result.data:
            raise RuntimeError(f"No student found with ID '{student_id}'.")
        return result.data[0]

    try:
        return _with_retry(_do_update)
    except Exception as e:
        raise RuntimeError(f"Failed to update student '{student_id}': {e}") from e


def delete_student(student_id: str) -> bool:
    """Delete a student and their attendance logs."""
    def _do_delete():
        client = get_supabase_client()
        # Delete attendance logs first (FK constraint)
        client.table("attendance_logs").delete().eq("student_id", student_id).execute()
        # Delete the student
        result = client.table("students").delete().eq("student_id", student_id).execute()
        if not result.data:
            raise RuntimeError(f"No student found with ID '{student_id}'.")
        return True

    try:
        return _with_retry(_do_delete)
    except Exception as e:
        raise RuntimeError(f"Failed to delete student '{student_id}': {e}") from e


def update_student_embedding(student_id: str, embedding: np.ndarray) -> Dict:
    """Re-register a student's face embedding (e.g., update photo)."""
    if embedding.shape != (512,):
        raise ValueError(f"Need 512D vector, got {embedding.shape}.")
    embedding_list = embedding.astype(np.float32).tolist()

    def _do_update():
        client = get_supabase_client()
        result = (
            client.table("students")
            .update({"embedding": embedding_list})
            .eq("student_id", student_id)
            .execute()
        )
        if not result.data:
            raise RuntimeError(f"No student found with ID '{student_id}'.")
        return result.data[0]

    try:
        return _with_retry(_do_update)
    except Exception as e:
        raise RuntimeError(f"Failed to update embedding for '{student_id}': {e}") from e

