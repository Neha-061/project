from fastapi import FastAPI
from routes import router
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import mediapipe as mp
import torch
import timm
from torchvision import transforms



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Life Cycle Started!")

    app.state.model = YOLO("backend/artifacts/yolov8n.pt")
    app.state.mp_face = mp.solutions.face_detection
    app.state.mp_hands = mp.solutions.hands

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_model = timm.create_model("efficientnet_b0", pretrained=True)
    eff_model.eval()
    eff_model.to(device)
    
    eff_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    app.state.eff_model = eff_model
    app.state.eff_transform = eff_transform
    app.state.device = device

    yield

    print("shutting down...")

    
app = FastAPI(lifespan = lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




app.include_router(router)
