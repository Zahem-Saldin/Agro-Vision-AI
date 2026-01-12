from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import motor.motor_asyncio
from bson import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

app = FastAPI(title="Crop Growth Prediction API")

# ---------- CORS ----------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MongoDB Connection ----------
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]
collections = {
    "paddy": db.paddy,
    "tea": db.tea,
    "coconut": db.coconut
}

# ---------- Pydantic Models ----------
class Prediction(BaseModel):
    year: int
    quarter: int
    dominant_phase: str
    soft_labels: List[float]
    indices: Dict[str, float]
    index_explanations: Dict[str, Dict[str, str | float]] | None = None
    image_base64: str | None = None
    productivity_color: List[int]
    productivity_score: float
    productivity_level: str

class SaveRequest(BaseModel):
    kml_text: str
    start_year: int
    end_year: int
    predictions: List[Prediction]

# ---------- Helper Functions ----------
async def save_predictions(collection, payload: SaveRequest, crop_type: str):
    doc = {
        "kml_text": payload.kml_text,
        "start_year": payload.start_year,
        "end_year": payload.end_year,
        "predictions": [p.dict() for p in payload.predictions],
        "crop_type": crop_type,
        "created_at": datetime.utcnow(),
    }
    result = await collection.insert_one(doc)
    return {"status": "success", "inserted_id": str(result.inserted_id)}

async def get_all_predictions_from(collection):
    docs = []
    cursor = collection.find()
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return {"status": "success", "data": docs}

async def delete_prediction(collection, doc_id: str):
    result = await collection.delete_one({"_id": ObjectId(doc_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "deleted_id": doc_id}
