from fastapi import APIRouter, UploadFile, File, Form, HTTPException,Request
import shutil
import os
import numpy as np
import cv2
from services import run_pipeline
import time


router = APIRouter()

UPLOAD_FOLDER = "backend/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/")
def root():
    return {"message": "ok"}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # future: YOLO model call here
    yolo_result = "YOLO will process this image later"

    return {
        "message": "Image uploaded successfully",
        "file": file.filename,
        "yolo": yolo_result
    }


@router.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = request.app.state.model
    mp_face = request.app.state.mp_face
    mp_hands = request.app.state.mp_hands

    req_id = str(int(time.time()*1000))
    save_dir = os.path.join(UPLOAD_FOLDER, req_id)
    result = run_pipeline(img, model, mp_face, mp_hands,save_dir)

    return result


@router.post("/ask")
async def analyze(request: Request,file: UploadFile = File(...)):

    contents = await file.read()
    req_id = str(int(time.time()*1000))
    save_dir = os.path.join(UPLOAD_FOLDER, req_id)
    os.makedirs(save_dir, exist_ok=True)
    f_name = file.filename

    """
    Uploading the main images and saving it
    """
    file_path = os.path.join(save_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    """
    Running the yolo model
    """
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = request.app.state.model
    mp_face = request.app.state.mp_face
    mp_hands = request.app.state.mp_hands
    result = run_pipeline(img, model, mp_face, mp_hands,save_dir)


    """
    Generation Effiecient Net score
    """

    return result
