from fastapi import FastAPI
from routes import router
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import mediapipe as mp



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Life Cycle Started!")

    app.state.model = YOLO("backend/artifacts/yolov8n.pt")
    app.state.mp_face = mp.solutions.face_detection
    app.state.mp_hands = mp.solutions.hands

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
