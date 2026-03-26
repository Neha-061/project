from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import os

router = APIRouter()
@router.get("/")
def root():
    return {"message": "ok"}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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