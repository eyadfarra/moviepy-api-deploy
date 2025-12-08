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
    font: Optional[str] = None  # Optional - uses system default if not provided
    fontsize: int = 60
    bg_color: Optional[str] = None


class Animation(BaseModel):
    enter_duration: float = 0.5
    exit_duration: float = 0.5
    float_amplitude: int = 10
    float_speed: float = 2
    effect: str = "fade"  # "fade" or "typing" (word-by-word reveal)


class HeaderLine(BaseModel):
    """A single line in the header."""
    text: str  # Required
    color: str = "white"
    bg_color: Optional[str] = None
    font: Optional[str] = None
    fontsize: int = 50


class HeaderConfig(BaseModel):
    """Header configuration with multiple lines and shared animation."""
    lines: List[HeaderLine]  # One or more lines
    animation: Optional[Animation] = None  # Shared animation for all lines


class RenderRequest(BaseModel):
    bg_video_url: str
    segments: List[Segment]
    main_style: TextStyle
    sub_style: TextStyle
    animation: Animation
    header: Optional[HeaderConfig] = None  # Optional header at top
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


def make_text_clip(text: str, font: str = None, fontsize: int = 60, bg_color: str = None):
    """Create a text clip with the given style."""
    # Handle font - use provided font, or fall back to system default
    if font:
        font_path = f"/fonts/{font}"  # Fonts mounted at /fonts
        if not os.path.exists(font_path):
            print(f"WARNING: Font {font} not found at {font_path}, using system default")
            font_path = None
    else:
        font_path = None
    
    # Build kwargs - only include bg_color if provided
    kwargs = {
        "text": text,
        "font_size": fontsize,
        "color": "white",
        "size": (TARGET_W - 80, None),
        "method": "caption",
        "margin": (20, 35)  # Increased margin for better visibility and full font height
    }
    
    # Only set font if we have a valid path
    if font_path:
        kwargs["font"] = font_path
    
    if bg_color:
        kwargs["bg_color"] = parse_color(bg_color)
    
    return TextClip(**kwargs)


def float_position(base_y, amp, speed):
    return lambda t: ("center", base_y + amp * math.sin(t * speed))


def create_typing_clips(text: str, style_kwargs: dict, start_time: float, 
                        total_duration: float, position_func, exit_duration: float = 0.5):
    """
    Create word-by-word typing animation clips.
    Returns a list of clips that when combined show words appearing progressively.
    """
    words = text.split()
    if not words:
        return []
    
    clips = []
    num_words = len(words)
    
    # Time allocated for typing (leave room for exit fade)
    typing_duration = total_duration * 0.7
    time_per_word = typing_duration / num_words if num_words > 0 else typing_duration
    
    for i, word in enumerate(words):
        # Build the text up to this word
        cumulative_text = " ".join(words[:i+1])
        
        # Word appears at this time
        word_start = start_time + (i * time_per_word)
        # Word stays visible until end of clip
        word_duration = (start_time + total_duration) - word_start
        
        if word_duration <= 0:
            continue
        
        # Create clip for this stage of typing
        word_kwargs = style_kwargs.copy()
        word_kwargs["text"] = cumulative_text
        
        clip = TextClip(**word_kwargs)
        clip = clip.with_start(word_start).with_duration(word_duration)
        clip = clip.with_position(position_func)
        
        # Add fade out only to the last word (the complete text)
        if i == num_words - 1:
            clip = clip.with_effects([vfx.CrossFadeOut(exit_duration)])
        
        clips.append(clip)
    
    return clips


def apply_blur(image):
    """Apply Gaussian blur to frame for text readability."""
    from PIL import Image, ImageFilter
    import numpy as np
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    # Apply Gaussian blur
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=5))
    return np.array(blurred)


def apply_effect(clip, effect):
    """
    Apply background effects to make white text more visible.
    
    Effects:
    - 'bw': Black and white
    - 'contrast': Increase contrast
    - 'darken': Darken the video (50% brightness) - good for white text
    - 'blur': Blur the background - makes text pop
    - 'overlay': Blur + darken combined - best for text visibility
    """
    e = effect.lower()
    
    if e == "bw":
        return clip.with_effects([vfx.BlackAndWhite()])
    
    if e == "contrast":
        return clip.with_effects([vfx.MultiplyColor(1.2)])
    
    if e == "darken":
        # Darken to 50% brightness for better text visibility
        return clip.with_effects([vfx.MultiplyColor(0.5)])
    
    if e == "blur":
        # Apply Gaussian blur using image_transform
        return clip.image_transform(apply_blur)
    
    if e == "overlay":
        # Combine blur and darken for maximum text visibility
        blurred = clip.image_transform(apply_blur)
        return blurred.with_effects([vfx.MultiplyColor(0.6)])
    
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

    # Resize first, then crop
    bg_resized = bg_raw.resized(height=TARGET_H)
    print(f"DEBUG: bg_resized size = {bg_resized.size}")
    
    bg = bg_resized.cropped(
        width=TARGET_W,
        height=TARGET_H,
        x_center=bg_resized.w/2,
        y_center=bg_resized.h/2
    )
    print(f"DEBUG: bg cropped size = {bg.size}")

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
    
    # Check for typing animation effect
    anim_effect = anim.get("effect", "fade")

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
                
                # Build style kwargs for text clip
                style_kwargs = {
                    "text": text,
                    "font_size": main_style.get("fontsize", 60),
                    "color": "white",
                    "size": (TARGET_W - 80, None),
                    "method": "caption",
                    "margin": (20, 35)
                }
                
                # Add font if available
                main_font = main_style.get("font")
                if main_font:
                    font_path = f"/fonts/{main_font}"
                    if os.path.exists(font_path):
                        style_kwargs["font"] = font_path
                
                # Add bg_color if set
                main_bg = main_style.get("bg_color")
                if main_bg:
                    style_kwargs["bg_color"] = parse_color(main_bg)
                
                position_func = float_position(TARGET_H * 0.42, float_amp, float_spd)
                
                if anim_effect == "typing":
                    # Use typing animation (word-by-word)
                    typing_clips = create_typing_clips(
                        text, style_kwargs, current_t, d, position_func, exit_dur
                    )
                    layers.extend(typing_clips)
                else:
                    # Use standard fade animation
                    clip = TextClip(**style_kwargs)
                    clip = clip.with_start(current_t).with_duration(d)
                    clip = clip.with_position(position_func).with_effects([
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
                    float_position(TARGET_H * 0.52, float_amp, float_spd)
                ).with_effects([
                    vfx.CrossFadeIn(enter_dur),
                    vfx.CrossFadeOut(exit_dur)
                ])
                layers.append(clip)
                current_t += d

    # --- HEADER (multi-line) ---
    header_config = request_data.get("header")
    if header_config and header_config.get("lines"):
        header_lines = header_config.get("lines", [])
        header_anim = header_config.get("animation")
        
        # Starting Y position (8% from top)
        y_offset = TARGET_H * 0.06
        line_spacing = 10  # pixels between lines
        
        for line_data in header_lines:
            line_text = line_data.get("text", "") if isinstance(line_data, dict) else str(line_data)
            if not line_text:
                continue
            
            # Build text clip kwargs
            line_kwargs = {
                "text": line_text,
                "font_size": line_data.get("fontsize", 50) if isinstance(line_data, dict) else 50,
                "color": line_data.get("color", "white") if isinstance(line_data, dict) else "white",
                "size": (TARGET_W - 80, None),
                "method": "caption",
                "margin": (15, 8)
            }
            
            # Optional font
            line_font = line_data.get("font") if isinstance(line_data, dict) else None
            if line_font:
                font_path = f"/fonts/{line_font}"
                if os.path.exists(font_path):
                    line_kwargs["font"] = font_path
            
            # Optional bg_color
            line_bg = line_data.get("bg_color") if isinstance(line_data, dict) else None
            if line_bg:
                line_kwargs["bg_color"] = parse_color(line_bg)
            
            line_clip = TextClip(**line_kwargs)
            line_clip = line_clip.with_start(0).with_duration(total_duration)
            line_clip = line_clip.with_position(("center", y_offset))
            
            # Apply shared animation if provided
            if header_anim:
                line_clip = line_clip.with_effects([
                    vfx.CrossFadeIn(header_anim.get("enter_duration", 0.5)),
                    vfx.CrossFadeOut(header_anim.get("exit_duration", 0.5))
                ])
            
            layers.append(line_clip)
            
            # Move Y down for next line (estimate line height based on fontsize)
            line_height = line_data.get("fontsize", 50) if isinstance(line_data, dict) else 50
            y_offset += line_height + line_spacing

    # Combine audio
    final_audio = concatenate_audioclips([info["audio"] for info in segment_infos])

    # Final composite - use explicit size since bg.size may be None in MoviePy 2.x
    print(f"DEBUG: Creating composite with size ({TARGET_W}, {TARGET_H})")
    final = CompositeVideoClip([bg] + layers, size=(TARGET_W, TARGET_H))
    final = final.with_duration(total_duration).with_audio(final_audio)

    # Write output to /data volume (mounted from host)
    os.makedirs("/data", exist_ok=True)
    filename = f"render_{uuid.uuid4()}.mp4"
    out = f"/data/{filename}"
    
    print(f"DEBUG: Writing video to: {out}")

    final.write_videofile(
        out,
        codec="libx264",
        audio_codec="aac",
        fps=request_data.get("fps", 30),
        logger="bar"
    )
    
    print(f"DEBUG: write_videofile completed")
    print(f"DEBUG: Expected output: {out}")
    print(f"DEBUG: File exists: {os.path.exists(out)}")
    
    # Search everywhere
    import glob
    import subprocess
    find_result = subprocess.run(["find", "/", "-name", "*.mp4", "-type", "f"], 
                                  capture_output=True, text=True, timeout=10)
    print(f"DEBUG: All mp4 files on system: {find_result.stdout.strip()}")

    # Search for render files in find output
    all_mp4 = find_result.stdout.strip().split('\n') if find_result.stdout.strip() else []
    print(f"DEBUG: All mp4 files on system: {all_mp4}")
    
    # Check if our expected file exists
    if not os.path.exists(out):
        # Try to find any render file
        for mp4 in all_mp4:
            if "render_" in mp4 and os.path.exists(mp4):
                print(f"DEBUG: Found render file at {mp4}")
                out = mp4
                break
    
    if not os.path.exists(out):
        # Last resort - use one of the temp mp4 files if it's our video
        # This shouldn't happen but MoviePy 2.x seems buggy
        raise Exception(f"Video rendering failed - file not found at {out}. All mp4: {all_mp4}")

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
        message=f"Job submitted successfully. Poll /status/{job_id} for updates."
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
