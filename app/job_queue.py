"""
Background job queue worker for video rendering.
"""
import os
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional
import json

from database import (
    init_db, 
    get_pending_jobs, 
    update_job_status, 
    get_expired_jobs,
    mark_job_expired,
    get_job
)


class JobQueue:
    """Background job queue manager."""
    
    def __init__(self, render_func: Callable):
        self.render_func = render_func
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the background workers."""
        if self.running:
            return
            
        self.running = True
        init_db()
        
        # Start job processing worker
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Start cleanup worker
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        print("JobQueue: Background workers started")
        
    def stop(self):
        """Stop the background workers."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        print("JobQueue: Background workers stopped")
    
    def _worker_loop(self):
        """Main worker loop that processes pending jobs."""
        import time
        while self.running:
            try:
                jobs = get_pending_jobs(limit=1)
                
                for job in jobs:
                    self._process_job(job)
                    
                # Sleep if no jobs found
                if not jobs:
                    time.sleep(2)  # Poll every 2 seconds
                    
            except Exception as e:
                print(f"JobQueue: Worker error: {e}")
                time.sleep(5)
    
    def _process_job(self, job: dict):
        """Process a single job."""
        job_id = job["id"]
        print(f"JobQueue: Processing job {job_id}")
        
        try:
            # Update status to processing
            update_job_status(job_id, "processing")
            
            # Parse request data
            request_data = json.loads(job["request_data"])
            
            # Call the render function
            video_path = self.render_func(request_data)
            
            # Calculate expiry time (10 minutes from now)
            expires_at = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
            
            # Update status to completed
            update_job_status(
                job_id, 
                "completed", 
                video_path=video_path,
                expires_at=expires_at
            )
            print(f"JobQueue: Job {job_id} completed, video at {video_path}")
            
        except Exception as e:
            print(f"JobQueue: Job {job_id} failed: {e}")
            import traceback
            traceback.print_exc()
            update_job_status(job_id, "failed", error_message=str(e))
    
    def _cleanup_loop(self):
        """Cleanup loop that deletes expired videos."""
        import time
        while self.running:
            try:
                expired_jobs = get_expired_jobs()
                
                for job in expired_jobs:
                    self._cleanup_job(job)
                
                # Check every 30 seconds
                time.sleep(30)
                
            except Exception as e:
                print(f"JobQueue: Cleanup error: {e}")
                time.sleep(60)
    
    def _cleanup_job(self, job: dict):
        """Clean up an expired job by deleting the video file."""
        job_id = job["id"]
        video_path = job.get("video_path")
        
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"JobQueue: Deleted expired video {video_path}")
            
            mark_job_expired(job_id)
            print(f"JobQueue: Marked job {job_id} as expired")
            
        except Exception as e:
            print(f"JobQueue: Cleanup failed for job {job_id}: {e}")


# Global job queue instance
job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global job_queue
    if job_queue is None:
        raise RuntimeError("Job queue not initialized")
    return job_queue


def init_job_queue(render_func: Callable) -> JobQueue:
    """Initialize the global job queue."""
    global job_queue
    job_queue = JobQueue(render_func)
    return job_queue
