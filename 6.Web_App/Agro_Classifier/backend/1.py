from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import rasterio
from skimage.transform import resize
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

# ------------------------ FastAPI Setup ------------------------
app = FastAPI(title="Crop Growth Prediction API")

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------ Models ------------------------
TARGET_SIZE = (128, 128)
epsilon = 1e-10

# -------------------- Local Yield Data (kg/ha or t/ha) --------------------
LOCAL_YIELD_DATA = {
    "paddy": {"min": 3.0, "max": 6.0},
    "tea": {"min": 1.0, "max": 2.5},
    "coconut": {"min": 3.0, "max": 7.0}
}

# ---- Indices Functions ----
def compute_indices_paddy(bands):
    B2, B3, B4, B8, B11 = [bands[b].astype(np.float32) for b in ['B2','B3','B4','B8','B11']]
    ndvi = (B8 - B4) / (B8 + B4 + epsilon)
    evi = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    savi = ((B8 - B4) * 1.5) / (B8 + B4 + 1.5 + epsilon)
    ndwi = (B8 - B11) / (B8 + B11 + epsilon)
    ndmi = (B8 - B11) / (B8 + B11 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)
    return [ndvi, evi, savi, ndwi, ndmi, gndvi]

def compute_indices_tea(bands):
    B2, B3, B4, B5, B8, B11 = [bands[b].astype(np.float32) for b in ['B2','B3','B4','B5','B8','B11']]
    ndvi = (B8 - B4) / (B8 + B4 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)
    evi = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    cigreen = (B8 / (B3 + epsilon)) - 1
    cire = (B8 / (B5 + epsilon)) - 1
    msi = B11 / (B8 + epsilon)
    return [ndvi, gndvi, evi, cigreen, cire, msi]

def compute_indices_coconut(bands):
    B2, B3, B4, B8, B11 = [bands[b].astype(np.float32) for b in ['B2','B3','B4','B8','B11']]
    ndvi = (B8 - B4) / (B8 + B4 + epsilon)
    evi = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)
    savi = ((B8 - B4) * 1.5) / (B8 + B4 + 1.5 + epsilon)
    ndmi = (B8 - B11) / (B8 + B11 + epsilon)
    msi = B11 / (B8 + epsilon)
    return [ndvi, evi, gndvi, savi, ndmi, msi]

# -------------------- Crop Config --------------------
CROP_CONFIG = {
    "paddy": {
        "bands": ['B2','B3','B4','B5','B8','B11'],
        "indices_fn": compute_indices_paddy,
        "model_path": "models/paddy_patch_based_softlabel_cnn_improved.keras",
        "phases": ["Sowing", "Vegetative", "Harvest"]
    },
    "tea": {
        "bands": ['B2','B3','B4','B5','B8','B11'],
        "indices_fn": compute_indices_tea,
        "model_path": "models/tea_patch_based_softlabel_cnn_improved.keras",
        "phases": ["Plucked", "Flush", "Mature"]
    },
    "coconut": {
        "bands": ['B2','B3','B4','B8','B11'],
        "indices_fn": compute_indices_coconut,
        "model_path": "models/coconut_patch_based_softlabel_cnn_improved.keras",
        "phases": ["Low", "Medium", "High"]
    },
}

PRODUCTIVITY_WEIGHTS = {
    "paddy": {
        "soft_labels": [0.2, 0.3, 0.5],
        "indices": {"ndvi": 0.15, "evi": 0.15}
    },
    "tea": {
        "soft_labels": [0.2, 0.2, 0.6],
        "indices": {"ndvi": 0.1, "evi": 0.1, "gndvi": 0.05}
    },
    "coconut": {
        "soft_labels": [0.2, 0.3, 0.5],
        "indices": {"ndvi": 0.1, "evi": 0.1, "gndvi": 0.05}
    },
}

# -------------------- Helpers --------------------
def safe_normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn)/(mx - mn + epsilon) if mx > mn else np.zeros_like(arr)

def compute_indices_safe(crop, bands_data):
    indices_fn = CROP_CONFIG[crop]["indices_fn"]
    return [np.nan_to_num(ind) for ind in indices_fn(bands_data)]

def prepare_cnn_input(indices_arr):
    img_stack = np.stack(indices_arr, axis=-1)
    for i in range(img_stack.shape[-1]):
        img_stack[..., i] = safe_normalize(img_stack[..., i])
    img_resized = resize(img_stack, (TARGET_SIZE[0], TARGET_SIZE[1], img_stack.shape[-1]),
                         mode='reflect', anti_aliasing=True)
    return img_resized.astype(np.float32)

def predict_growth_phase(img_array, crop):
    model = CROP_CONFIG[crop]["model"]
    phases = CROP_CONFIG[crop]["phases"]
    pred = model.predict(np.expand_dims(img_array, axis=0))
    pred = np.ravel(pred)
    if pred.size != len(phases):
        pred = np.mean(pred.reshape(-1, len(phases)), axis=0)
    return {"soft_labels": pred.tolist(), "dominant_phase": phases[np.argmax(pred)]}

# ------------------------ Productivity Scoring ------------------------
def compute_productivity_score(crop, soft_labels, indices_dict, local_yield=None):
    weights = PRODUCTIVITY_WEIGHTS[crop]
    sl_weights = weights["soft_labels"]
    idx_weights = weights["indices"]

    # Weighted soft-label score
    sl_score = sum(soft_labels[i] * sl_weights[i] for i in range(len(soft_labels)))
    idx_score = sum(indices_dict.get(k, 0) * idx_weights[k] for k in idx_weights)

    # Include normalized local yield if provided
    yield_score = 0
    if local_yield is not None:
        min_y, max_y = LOCAL_YIELD_DATA[crop]["min"], LOCAL_YIELD_DATA[crop]["max"]
        yield_score = np.clip((local_yield - min_y) / (max_y - min_y + epsilon), 0, 1)

    total_raw = sl_score + idx_score + yield_score

    # Normalize 0-100
    score = float(np.clip(total_raw * 100 / (1 + (yield_score>0)), 0, 100))

    # Level
    if score >= 50:
        level = "High"
    elif score >= 30:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level}

def productivity_color(crop: str, soft_labels, indices, local_yield=None):
    prod = compute_productivity_score(crop, soft_labels, indices, local_yield)
    score = prod["score"]

    # Red → Yellow → Green gradient
    if score <= 50:
        r, g, b = 255, int((score / 50) * 255), 0
    else:
        r, g, b = int((1 - (score - 50) / 50) * 255), 255, 0

    return {"rgb": [r, g, b], "score": score, "level": prod["level"]}

# -------------------- Endpoint --------------------
def make_tiff_predict_endpoint(crop: str):
    async def endpoint(files: list[UploadFile] = File(...), local_yield: float = Form(None)):
        results = []
        config = CROP_CONFIG[crop]
        indices_keys_map = {
            "paddy": ["NDVI","EVI","SAVI","NDWI","NDMI","GNDVI"],
            "tea": ["NDVI","GNDVI","EVI","CIGREEN","CIRE","MSI"],
            "coconut": ["NDVI","EVI","GNDVI","SAVI","NDMI","MSI"]
        }
        indices_keys = indices_keys_map[crop]

        for f in files:
            try:
                with rasterio.open(f.file) as src:
                    bands_data = {}
                    for i, band in enumerate(config["bands"]):
                        arr = src.read(i+1).astype(np.float32)
                        arr[arr < 0] = 0
                        arr /= arr.max() if arr.max() > 1 else 1.0
                        bands_data[band] = arr

                    # Compute indices and CNN input
                    indices_arr = compute_indices_safe(crop, bands_data)
                    cnn_input = prepare_cnn_input(indices_arr)
                    pred = predict_growth_phase(cnn_input, crop)

                    # Map indices for display
                    indices_dict = {k: float(indices_arr[i].mean()) for i,k in enumerate(indices_keys[:len(indices_arr)])}

                    # Productivity
                    prod_color = productivity_color(crop, pred["soft_labels"], indices_dict, local_yield)

                    # RGB preview
                    if all(b in bands_data for b in ["B4","B3","B2"]):
                        rgb_array = np.stack([bands_data['B4'], bands_data['B3'], bands_data['B2']], axis=-1)
                    else:
                        rgb_array = np.stack([bands_data[config["bands"][i]] for i in range(3)], axis=-1)

                    rgb_norm = ((rgb_array - rgb_array.min())/(rgb_array.max()-rgb_array.min()+epsilon)*255).astype(np.uint8)
                    pil_img = Image.fromarray(rgb_norm)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

                    results.append({
                        "filename": f.filename,
                        **pred,
                        "indices": indices_dict,
                        "productivity_score": prod_color["score"],
                        "productivity_level": prod_color["level"],
                        "productivity_color": prod_color["rgb"],
                        "image_base64": img_base64,
                        "note": None,
                        "local_yield_used": local_yield
                    })
            except Exception as e:
                results.append({"filename": f.filename, "error": str(e)})
        return results
    return endpoint

# -------------------- Register Endpoints --------------------
app.post("/predict_paddy_image")(make_tiff_predict_endpoint("paddy"))
app.post("/predict_tea_image")(make_tiff_predict_endpoint("tea"))
app.post("/predict_coconut_image")(make_tiff_predict_endpoint("coconut"))

# -------------------- Load Models at Startup --------------------
for crop, config in CROP_CONFIG.items():
    config["model"] = load_model(config["model_path"])
