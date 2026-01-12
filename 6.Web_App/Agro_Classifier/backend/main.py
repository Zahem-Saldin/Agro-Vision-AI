from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib

# Import apps
app1 = importlib.import_module("1").app
app2 = importlib.import_module("2").app
app3 = importlib.import_module("3").app

# Unified app
main_app = FastAPI(title="Crop Growth Prediction - Unified API")

# Enable CORS globally
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-apps
main_app.mount("/tiff", app1)
main_app.mount("/kml", app2)
main_app.mount("/save", app3)

@main_app.get("/")
def root():
    return {
        "message": "Unified Crop Growth Prediction API",
        "endpoints": {
            "TIFF API": "/tiff",
            "KML API": "/kml",
            "Save API": "/save"
        }
    }
