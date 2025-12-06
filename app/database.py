"""
SQLite database setup for job queue management.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

DATABASE_PATH = "/tmp/jobs.db"


def init_db():
    """Initialize the database and create tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                video_path TEXT,
                error_message TEXT,
                request_data TEXT NOT NULL,
                expires_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_expires_at ON jobs(expires_at)
        """)
        conn.commit()


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_job(job_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new job in the database."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, status, created_at, request_data)
            VALUES (?, 'pending', ?, ?)
            """,
            (job_id, now, json.dumps(request_data))
        )
        conn.commit()
    return {"id": job_id, "status": "pending", "created_at": now}


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None


def get_pending_jobs(limit: int = 10) -> list:
    """Get pending jobs for processing."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs 
            WHERE status = 'pending' 
            ORDER BY created_at ASC 
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def update_job_status(
    job_id: str, 
    status: str, 
    video_path: Optional[str] = None,
    error_message: Optional[str] = None,
    expires_at: Optional[str] = None
):
    """Update job status."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        if status == "processing":
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                (status, now, job_id)
            )
        elif status == "completed":
            conn.execute(
                """
                UPDATE jobs 
                SET status = ?, completed_at = ?, video_path = ?, expires_at = ?
                WHERE id = ?
                """,
                (status, now, video_path, expires_at, job_id)
            )
        elif status == "failed":
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, error_message = ? WHERE id = ?",
                (status, now, error_message, job_id)
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                (status, job_id)
            )
        conn.commit()


def get_expired_jobs() -> list:
    """Get jobs that have expired and need cleanup."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs 
            WHERE status = 'completed' 
            AND expires_at IS NOT NULL 
            AND expires_at < ?
            """,
            (now,)
        ).fetchall()
        return [dict(row) for row in rows]


def mark_job_expired(job_id: str):
    """Mark a job as expired after cleanup."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'expired', video_path = NULL WHERE id = ?",
            (job_id,)
        )
        conn.commit()
