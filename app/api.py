from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import threading
import uvicorn
import shutil
import os
import json
from roop import globals as roop_globals
from roop import utilities as util
from roop.ProcessEntry import ProcessEntry
from roop.core import batch_process_regular
import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_TEMP = os.path.join(os.getcwd(), "temp", "api_uploads")
os.makedirs(API_TEMP, exist_ok=True)

@app.get("/api/settings")
def get_settings():
    if roop_globals.CFG:
        return roop_globals.CFG.__dict__
    return {}

@app.post("/api/settings")
def save_settings(settings: dict):
    if roop_globals.CFG:
        for k, v in settings.items():
            if hasattr(roop_globals.CFG, k):
                setattr(roop_globals.CFG, k, v)
        roop_globals.CFG.save()
    return {"status": "success"}

@app.post("/api/upload_source")
async def upload_source(file: UploadFile = File(...)):
    path = os.path.join(API_TEMP, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Normally roop uses get_face_images and FaceSet
    # For now, just return the path to it
    return {"path": path}

@app.post("/api/upload_target")
async def upload_target(file: UploadFile = File(...)):
    path = os.path.join(API_TEMP, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"path": path}

@app.post("/api/swap")
async def trigger_swap(payload: dict, background_tasks: BackgroundTasks):
    source_path = payload.get("source_path")
    target_path = payload.get("target_path")
    
    if not source_path or not target_path:
        return JSONResponse(status_code=400, content={"message": "Missing source or target"})

    # Setup globals just like start_swap does
    roop_globals.target_path = target_path
    
    # ProcessEntry is what batch_process_regular expects
    list_files_process = [ProcessEntry(target_path, 0, 0, 0)]
    
    from roop.capturer import get_video_frame_total
    if util.is_video(target_path) or target_path.lower().endswith('gif') or util.is_animated_webp(target_path):
        frames = get_video_frame_total(target_path)
        if frames:
            list_files_process[0].endframe = frames

    # Create dummy progress
    class DummyProgress:
        def __call__(self, progress, desc=""):
            print(f"Progress: {progress} {desc}")
    
    def process_task():
        # Setup inputs
        import ui.globals
        ui.globals.ui_input_thumbs = [source_path] 
        roop_globals.INPUT_FACESETS = [[]]
        from roop.face_util import extract_face_images
        faces = extract_face_images(source_path, False)
        if faces:
            roop_globals.INPUT_FACESETS = [faces]
        else:
            return

        batch_process_regular(
            "Extract & Re-encode", list_files_process, "None", "", False, None, False, 1, DummyProgress(), 0,
            use_3d_recon=False, mask_per_frame_json="", use_source_bank=False,
            use_frontalization=False, frontalization_threshold=25.0, swap_model='inswapper'
        )

    background_tasks.add_task(process_task)
    return {"status": "started"}

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")
