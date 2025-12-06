import os, math, tempfile, requests, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_audioclips,
    vfx
)
from database import init_db, create_job, get_job
from job_queue import init_job_queue, get_job_queue

TARGET_W, TARGET_H = 1080, 1920


# ---------- MODELS ----------
class TextItem(BaseModel):
    text: str
    duration: Optional[float] = None


class Segment(BaseModel):
    audio_url: str
    main_texts: List[TextItem]
    sub_texts: Optional[List[TextItem]] = []


class TextStyle(BaseModel):
    font: str
    fontsize: int
    bg_color: str


class Animation(BaseModel):
    enter_duration: float = 0.5
    exit_duration: float = 0.5
    float_amplitude: int = 10
    float_speed: float = 2


class RenderRequest(BaseModel):
    bg_video_url: str
    segments: List[Segment]
    main_style: TextStyle
    sub_style: TextStyle
    animation: Animation
    effect: str = "none"
    fps: int = 30


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


# ---------- APP LIFECYCLE ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("Starting application...")
    init_db()
    queue = init_job_queue(process_render_job)
    queue.start()
    yield
    # Shutdown
    print("Shutting down...")
    queue.stop()


app = FastAPI(lifespan=lifespan)


# ---------- HELPERS ----------
def download_file(url: str, suffix: str):
    """Download file from URL to temp path."""
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise Exception(f"Failed to download {url}: {e}")

    fd, p = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for c in r.iter_content(8192):
            f.write(c)
    return p


def parse_color(color_str: str):
    """Convert CSS rgba() or other color formats to Pillow-compatible format."""
    if color_str.startswith("rgba("):
        content = color_str[5:-1]
        parts = [p.strip() for p in content.split(",")]
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a = int(float(parts[3]) * 255)
        return (r, g, b, a)
    elif color_str.startswith("rgb("):
        content = color_str[4:-1]
        parts = [p.strip() for p in content.split(",")]
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    return color_str


def make_text_clip(text: str, font: str, fontsize: int, bg_color: str):
    """Create a text clip with the given style."""
    font_path = f"/app/fonts/{font}"
    if not os.path.exists(font_path):
        raise Exception(f"Font not found: {font}")

    return TextClip(
        text=text,
        font_size=fontsize,
        font=font_path,
        color="white",
        bg_color=parse_color(bg_color),
        size=(TARGET_W - 100, None),
        method="caption"
    )


def float_position(base_y, amp, speed):
    return lambda t: ("center", base_y + amp * math.sin(t * speed))


def apply_effect(clip, effect):
    e = effect.lower()
    if e == "bw":
        return clip.with_effects([vfx.BlackAndWhite()])
    if e == "contrast":
        return clip.with_effects([vfx.MultiplyColor(1.2)])
    return clip


# ---------- RENDER PROCESSING ----------
def process_render_job(request_data: dict) -> str:
    """
    Process a render job and return the video path.
    This runs in a background thread.
    """
    # Download background video
    bg_path = download_file(request_data["bg_video_url"], ".mp4")
    bg_raw = VideoFileClip(bg_path)

    bg = bg_raw.resized(height=TARGET_H).cropped(
        width=TARGET_W,
        height=TARGET_H,
        x_center=bg_raw.w/2,
        y_center=bg_raw.h/2
    )

    segment_infos = []
    global_t = 0.0

    for seg in request_data["segments"]:
        audio_path = download_file(seg["audio_url"], ".mp3")
        audio_clip = AudioFileClip(audio_path)
        dur = audio_clip.duration

        segment_infos.append({
            "audio": audio_clip,
            "main": seg.get("main_texts", []),
            "sub": seg.get("sub_texts", []),
            "start": global_t,
            "duration": dur
        })
        global_t += dur

    total_duration = global_t

    # Loop background
    loops = math.ceil(total_duration / bg.duration)
    bg = bg.with_effects([vfx.Loop(n=loops)]).subclipped(0, total_duration)
    bg = apply_effect(bg, request_data.get("effect", "none"))

    # Build layers
    layers = []
    anim = request_data.get("animation", {})
    enter_dur = anim.get("enter_duration", 0.5)
    exit_dur = anim.get("exit_duration", 0.5)
    float_amp = anim.get("float_amplitude", 10)
    float_spd = anim.get("float_speed", 2)

    main_style = request_data["main_style"]
    sub_style = request_data["sub_style"]

    for seg in segment_infos:
        seg_start = seg["start"]
        seg_dur = seg["duration"]

        # Main texts
        if seg["main"]:
            auto_dur = seg_dur / len(seg["main"])
            current_t = seg_start
            for item in seg["main"]:
                text = item.get("text", "") if isinstance(item, dict) else item
                d = (item.get("duration") if isinstance(item, dict) else None) or auto_dur
                
                clip = make_text_clip(
                    text,
                    main_style["font"],
                    main_style["fontsize"],
                    main_style["bg_color"]
                )
                clip = clip.with_start(current_t).with_duration(d)
                clip = clip.with_position(
                    float_position(TARGET_H * 0.38, float_amp, float_spd)
                ).with_effects([
                    vfx.CrossFadeIn(enter_dur),
                    vfx.CrossFadeOut(exit_dur)
                ])
                layers.append(clip)
                current_t += d

        # Sub texts
        if seg["sub"]:
            auto_dur = seg_dur / len(seg["sub"])
            current_t = seg_start
            for item in seg["sub"]:
                text = item.get("text", "") if isinstance(item, dict) else item
                d = (item.get("duration") if isinstance(item, dict) else None) or auto_dur
                
                clip = make_text_clip(
                    text,
                    sub_style["font"],
                    sub_style["fontsize"],
                    sub_style["bg_color"]
                )
                clip = clip.with_start(current_t).with_duration(d)
                clip = clip.with_position(
                    float_position(TARGET_H * 0.55, float_amp, float_spd)
                ).with_effects([
                    vfx.CrossFadeIn(enter_dur),
                    vfx.CrossFadeOut(exit_dur)
                ])
                layers.append(clip)
                current_t += d

    # Combine audio
    final_audio = concatenate_audioclips([info["audio"] for info in segment_infos])

    # Final composite
    final = CompositeVideoClip([bg] + layers, size=(TARGET_W, TARGET_H))
    final = final.with_duration(total_duration).with_audio(final_audio)

    # Write output
    filename = f"render_{uuid.uuid4()}.mp4"
    out = f"/tmp/{filename}"

    final.write_videofile(
        out,
        codec="libx264",
        audio_codec="aac",
        fps=request_data.get("fps", 30),
        logger="bar"
    )

    if not os.path.exists(out):
        raise Exception("Video rendering failed - output file not created")

    # Cleanup temp files
    try:
        os.remove(bg_path)
        for seg in segment_infos:
            if hasattr(seg["audio"], 'filename'):
                os.remove(seg["audio"].filename)
    except:
        pass

    return out


# ---------- API ENDPOINTS ----------
@app.post("/render", response_model=JobResponse)
def submit_render(req: RenderRequest):
    """Submit a render job and return job ID immediately."""
    job_id = str(uuid.uuid4())
    
    # Convert Pydantic model to dict for storage
    request_data = req.model_dump()
    
    # Create job in database
    create_job(job_id, request_data)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully. Poll /status/{job_id} for updates."
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    """Get the status of a render job."""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    
    response = StatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at")
    )
    
    if job["status"] == "completed" and job.get("video_path"):
        response.video_url = f"/download/{job_id}"
    elif job["status"] == "failed":
        response.error = job.get("error_message")
    elif job["status"] == "expired":
        response.error = "Video has expired and been deleted"
    
    return response


@app.get("/download/{job_id}")
def download_video(job_id: str):
    """Download the completed video for a job."""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    
    if job["status"] != "completed":
        raise HTTPException(400, f"Job is not completed. Current status: {job['status']}")
    
    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(410, "Video file has expired or been deleted")
    
    return FileResponse(
        video_path, 
        media_type="video/mp4", 
        filename="short.mp4"
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
