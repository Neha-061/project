from fastapi import APIRouter, UploadFile, File, Form, HTTPException,Request
import shutil
import os
import numpy as np
import cv2
import time
from typing import List
from PIL import Image
import io
from services import run_pipeline, get_image_score



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



@router.post("/score")
async def score_images(
    request: Request,
    original: UploadFile = File(...),
    crops: List[UploadFile] = File(...)
):
    try:
        eff_model = request.app.state.eff_model
        eff_transform = request.app.state.eff_transform
        device = request.app.state.device

        # -------- ORIGINAL IMAGE --------
        original_bytes = await original.read()
        original_image = Image.open(io.BytesIO(original_bytes)).convert("RGB")

        original_score = get_image_score(original_image, eff_model, eff_transform, device)

        # -------- CROPPED IMAGES --------
        crop_scores = []

        for crop in crops:
            crop_bytes = await crop.read()
            crop_image = Image.open(io.BytesIO(crop_bytes)).convert("RGB")

            score = get_image_score(crop_image, eff_model, eff_transform, device)
            crop_scores.append(score)

        # -------- AVERAGE --------
        avg_crop_score = sum(crop_scores) / len(crop_scores) if crop_scores else 0

        # -------- FINAL SCORE --------
        final_score = (0.6 * original_score) + (0.4 * avg_crop_score)

        return {
            "original_score": original_score,
            "crop_scores": crop_scores,
            "average_crop_score": avg_crop_score,
            "final_score": final_score
        }

    except Exception as e:
        return {"error": str(e)}
