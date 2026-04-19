from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="ML Model API")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

from app.core.model import load_model

@app.on_event("startup")
def startup_event():
    load_model()
    print("✅ Model loaded at startup")