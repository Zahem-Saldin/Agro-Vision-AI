from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ee
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
import base64
import requests
import json
import time

# ------------------------ FastAPI Setup ------------------------
app = FastAPI(title="Crop Growth Prediction API")

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------ Initialize EE ------------------------
ee.Initialize(project='skepter2k')
sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .filter(ee.Filter.lt('MEAN_SOLAR_ZENITH_ANGLE', 80))

# ------------------------ Models ------------------------
TARGET_SIZE = (128, 128)
epsilon = 1e-10

MODELS = {
    "paddy": {
        "path": "models/paddy_patch_based_softlabel_cnn_improved.keras",
        "phases": ["Sowing", "Vegetative", "Harvest"],
    },
    "tea": {
        "path": "models/tea_patch_based_softlabel_cnn_improved.keras",
        "phases": ["Plucked", "Flush", "Mature"],
    },
    "coconut": {
        "path": "models/coconut_patch_based_softlabel_cnn_improved.keras",
        "phases": ["Low", "Medium", "High"],
    },
}

PRODUCTIVITY_WEIGHTS = {
    "paddy": {
        # Model phases: ["Sowing", "Vegetative", "Harvest"]
        "soft_labels": [0.2, 0.3, 0.5],  # Sowing, Vegetative, Harvest importance
        "indices": {"ndvi": 0.05, "evi": 0.05},  # remaining 0.1
    },
    "tea": {
        # Model phases: ["Plucked", "Flush", "Mature"]
        "soft_labels": [0.2, 0.2, 0.6],  # Plucked, Flush, Mature importance
        "indices": {"ndvi": 0.05, "evi": 0.05},
    },
    "coconut": {
        # Model phases: ["Low", "Medium", "High"]
        "soft_labels": [0.2, 0.3, 0.5],  # Low, Medium, High importance
        "indices": {"ndvi": 0.05, "evi": 0.05},
    },
}

INDEX_EXPLANATIONS = {
    "NDVI": {
        "desc": "Normalized Difference Vegetation Index measures vegetation greenness.",
        "interpretation": lambda v: "Positive (healthy vegetation)" if v > 0.5 else "Negative (sparse vegetation)"
    },
    "EVI": {
        "desc": "Enhanced Vegetation Index captures vegetation density and canopy structure.",
        "interpretation": lambda v: "Positive (dense green)" if v > 1 else "Negative (less dense)"
    },
    "SAVI": {
        "desc": "Soil Adjusted Vegetation Index reduces soil brightness effects on vegetation signal.",
        "interpretation": lambda v: "Positive (healthy plants)" if v > 0.5 else "Negative (stressed plants)"
    },
    "NDWI": {
        "desc": "Normalized Difference Water Index indicates surface water content.",
        "interpretation": lambda v: "Positive (water present)" if v > 0 else "Negative (dry soil)"
    },
    "NDMI": {
        "desc": "Normalized Difference Moisture Index shows vegetation water content.",
        "interpretation": lambda v: "Positive (well-watered)" if v > 0 else "Negative (low moisture)"
    },
    "GNDVI": {
        "desc": "Green NDVI measures chlorophyll content using green band.",
        "interpretation": lambda v: "Positive (high chlorophyll)" if v > 0.5 else "Negative (low chlorophyll)"
    },
    "CIGREEN": {
        "desc": "Chlorophyll Index Green estimates chlorophyll concentration.",
        "interpretation": lambda v: "Positive (high chlorophyll)" if v > 1 else "Negative (low chlorophyll)"
    },
    "CIRE": {
        "desc": "Chlorophyll Index Red Edge estimates chlorophyll using red edge band.",
        "interpretation": lambda v: "Positive (high chlorophyll)" if v > 1 else "Negative (low chlorophyll)"
    },
    "MSI": {
        "desc": "Moisture Stress Index indicates plant stress from water deficiency.",
        "interpretation": lambda v: "Positive (low stress)" if v < 0.6 else "Negative (high stress)"
    }
}

for crop in MODELS:
    MODELS[crop]["model"] = load_model(MODELS[crop]["path"])

# ------------------------ Request Schema ------------------------
class PredictionRequest(BaseModel):
    kml_text: str
    start_year: int
    end_year: int

# ------------------------ Helper Functions ------------------------
def parse_kml(kml_text: str):

    start = kml_text.find('<kml')
    end   = kml_text.rfind('</kml>') + len('</kml>')
    kml_clean = kml_text[start:end]

    root = ET.fromstring(kml_clean)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    polygons = []

    for coords_node in root.findall('.//kml:coordinates', ns):
        coords = []
        for line in coords_node.text.strip().split():
            parts = line.split(',')

            lon, lat = map(float, parts[:2])
            coords.append((lon, lat))
        # Ensure polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygons.append(coords)

    if not polygons:
        raise ValueError("No polygon coordinates found in KML")


    return polygons, ee.Geometry.MultiPolygon([polygons])


def mask_clouds(image):
    scl = image.select('SCL')
    cloud_shadow = scl.eq(3)
    clouds = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
    mask = clouds.Or(cloud_shadow).Not()
    return image.updateMask(mask)

def usable_fraction(image, roi):
    scl = image.select('SCL')
    cloud_shadow = scl.eq(3)
    clouds = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
    mask = clouds.Or(cloud_shadow).Not()
    stats = mask.reduceRegion(reducer=ee.Reducer.mean(),
                              geometry=roi, scale=10, maxPixels=1e9)
    return stats.get('SCL')

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

CROP_CONFIG = {
    "paddy": {"bands": ['B2','B3','B4','B8','B11'], "indices_fn": compute_indices_paddy},
    "tea": {"bands": ['B2','B3','B4','B5','B8','B11'], "indices_fn": compute_indices_tea},
    "coconut": {"bands": ['B2','B3','B4','B8','B11'], "indices_fn": compute_indices_coconut},
}

def ee_image_to_array(image, roi, crop: str):
    config = CROP_CONFIG[crop]
    img_dict = image.select(config["bands"]).reduceRegion(
        reducer=ee.Reducer.toList(), geometry=roi, scale=10, maxPixels=1e9).getInfo()
    bands = {b: np.array(img_dict.get(b, [0]), dtype=np.float32) for b in config["bands"]}
    indices = [np.nan_to_num(ind) for ind in config["indices_fn"](bands)]
    img_stack = np.stack(indices, axis=-1)
    for i in range(img_stack.shape[-1]):
        band_min, band_max = np.min(img_stack[..., i]), np.max(img_stack[..., i])
        img_stack[..., i] = (img_stack[..., i] - band_min) / (band_max - band_min + epsilon) if band_max > band_min else 0.0
    img_resized = resize(img_stack, (TARGET_SIZE[0], TARGET_SIZE[1], img_stack.shape[-1]),
                         mode='reflect', anti_aliasing=True)
    return img_resized.astype(np.float32)

def predict_growth_phase(img_array, crop: str):
    model = MODELS[crop]["model"]
    phases = MODELS[crop]["phases"]
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    prediction = np.ravel(prediction)
    if prediction.size != 3:
        prediction = np.mean(prediction.reshape(-1, 3), axis=0)
    dominant_phase = phases[np.argmax(prediction)]
    return {"soft_labels": prediction.tolist(), "dominant_phase": dominant_phase}

def ee_image_to_base64_visualize(image, roi):
    viz = image.visualize(bands=['B4','B3','B2'], min=0, max=3000)
    url = viz.getThumbURL({'region': roi, 'format':'png', 'dimensions':256})
    resp = requests.get(url)
    return f"data:image/png;base64,{base64.b64encode(resp.content).decode()}"

def get_quarter_images(roi, year, start_month):
    sd = ee.Date.fromYMD(year, start_month, 1)
    ed = sd.advance(3, 'month')
    return sentinel.filterBounds(roi).filterDate(sd, ed)

def get_indices_keys_for_crop(crop: str):
    if crop == "paddy":
        return ["ndvi","evi","savi","ndwi","ndmi","gndvi"]
    elif crop == "tea":
        return ["ndvi","gndvi","evi","cigreen","cire","msi"]
    elif crop == "coconut":
        return ["ndvi","evi","gndvi","savi","ndmi","msi"]
    else:
        return []

# ------------------------ Productivity Scoring ------------------------
def compute_productivity_score(crop, soft_labels, indices_dict):
    weights = PRODUCTIVITY_WEIGHTS[crop]
    sl_weights = weights["soft_labels"]
    idx_weights = weights["indices"]

    # Weighted sum of soft labels
    sl_score = sum(soft_labels[i] * 100 * sl_weights[i] for i in range(len(soft_labels)))

    # Weighted sum of indices (already normalized 0-1)
    idx_score = sum(indices_dict.get(k, 0) * 100 * w for k, w in idx_weights.items())

    # Total productivity score
    score = sl_score + idx_score

    if score >= 50:
        level = "High"
    elif score >= 30:
        level = "Medium"
    else:
        level = "Low"

    return {"score": float(score), "level": level}

# ------------------------ Productivity Map ------------------------
def productivity_color(crop: str, soft_labels, indices):
    prod = compute_productivity_score(crop, soft_labels, indices)
    score = prod["score"]  # 0-100

    # Red -> Yellow -> Green gradient
    if score <= 50:
        # Red to Yellow
        r = 255
        g = int((score / 50) * 255)
        b = 0
    else:
        # Yellow to Green
        r = int((1 - (score - 50) / 50) * 255)
        g = 255
        b = 0

    return {"rgb": [r, g, b], "score": score, "level": prod["level"]}


# ------------------------ Generic Endpoint ------------------------
def make_predict_endpoint(crop: str):
    async def endpoint(request: PredictionRequest):
        def result_generator():
            try:
                coords, roi = parse_kml(request.kml_text)
                yield "["
                first_item = True

                yearly_data = {}

                for year in range(request.start_year, request.end_year + 1):
                    yearly_data[year] = {"soft_labels_sum": None, "prod_score_sum": 0, "quarters_count": 0, "dominant_phase_count": {}}

                    for start_month in [1, 4, 7, 10]:
                        quarter = ((start_month - 1) // 3) + 1
                        fallback_years = [year, year-1, year+1, year-2, year+2, year-3, year+3, year-4, year+4]
                        best_image = None

                        for candidate_year in fallback_years:
                            filtered = get_quarter_images(roi, candidate_year, start_month)
                            if filtered.size().getInfo() == 0:
                                continue
                            candidate_img = filtered.sort('CLOUDY_PIXEL_PERCENTAGE').first()
                            try:
                                fraction = usable_fraction(candidate_img, roi).getInfo()
                            except:
                                continue
                            if fraction is not None and fraction >= 0.7:
                                best_image = candidate_img
                                break

                        if best_image is None:
                            result = {
                                "year": year, "quarter": quarter,
                                "dominant_phase": None, "soft_labels": None,
                                "image_base64": None, "indices": None,
                                "productivity_score": None, "productivity_level": None,
                                "productivity_map": None,
                                "note": "No suitable image found"
                            }
                        else:
                            try:
                                masked_image = mask_clouds(best_image)
                                cnn_array = ee_image_to_array(masked_image, roi, crop)
                                pred = predict_growth_phase(cnn_array, crop)
                                img_base64 = ee_image_to_base64_visualize(masked_image, roi)

                                config = CROP_CONFIG[crop]
                                img_dict = masked_image.select(config["bands"]).reduceRegion(
                                    reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9
                                ).getInfo()
                                bands = {b: np.array(img_dict.get(b, [0]), dtype=np.float32) for b in config["bands"]}
                                indices_vals = [np.nan_to_num(ind) for ind in config["indices_fn"](bands)]
                                indices_keys = get_indices_keys_for_crop(crop)
                                indices_dict = {k: float(indices_vals[i]) for i, k in enumerate(indices_keys[:len(indices_vals)])}

                                prod = compute_productivity_score(crop, pred["soft_labels"], indices_dict)
                                prod_color = productivity_color(crop, pred["soft_labels"], indices_dict)

                                explanations = {}
                                for k, v in indices_dict.items():
                                    info = INDEX_EXPLANATIONS.get(k.upper())
                                    if info:
                                        explanations[k.upper()] = {
                                            "value": v,
                                            "effect": info["interpretation"](v),
                                            "description": info["desc"]
                                        }

                                result = {
                                    "year": year, "quarter": quarter,
                                    **pred,
                                    "image_base64": img_base64,
                                    "indices": indices_dict,
                                    "index_explanations": explanations,
                                    "productivity_score": prod_color["score"],
                                    "productivity_level": prod_color["level"],
                                    "productivity_color": prod_color["rgb"],
                                    "note": None
                                }

                                ydata = yearly_data[year]
                                if ydata["soft_labels_sum"] is None:
                                    ydata["soft_labels_sum"] = np.array(pred["soft_labels"])
                                else:
                                    ydata["soft_labels_sum"] += np.array(pred["soft_labels"])
                                ydata["prod_score_sum"] += prod["score"]
                                ydata["quarters_count"] += 1
                                phase = pred["dominant_phase"]
                                if phase:
                                    ydata["dominant_phase_count"][phase] = ydata["dominant_phase_count"].get(phase, 0) + 1

                            except Exception as e:
                                result = {
                                    "year": year, "quarter": quarter,
                                    "dominant_phase": None, "soft_labels": None,
                                    "image_base64": None, "indices": None,
                                    "productivity_score": None, "productivity_level": None,
                                    "productivity_map": None,
                                    "note": str(e)
                                }

                        if not first_item:
                            yield ","
                        yield json.dumps(result)
                        first_item = False
                        time.sleep(0.1)

                # Yearly summaries
                for year in range(request.start_year, request.end_year + 1):
                    ydata = yearly_data[year]
                    if ydata["quarters_count"] == 0:
                        continue
                    avg_soft_labels = (ydata["soft_labels_sum"] / ydata["quarters_count"]).tolist()
                    avg_prod_score = ydata["prod_score_sum"] / ydata["quarters_count"]
                    dominant_phase_pct = {phase: count / ydata["quarters_count"] * 100
                                          for phase, count in ydata["dominant_phase_count"].items()}

                    # Compute yearly productivity level
                    if avg_prod_score >= 60:
                        yearly_level = "High"
                    elif avg_prod_score >= 40:
                        yearly_level = "Medium"
                    else:
                        yearly_level = "Low"

                    summary = {
                        "year": year,
                        "yearly_avg_soft_labels": avg_soft_labels,
                        "yearly_avg_productivity_score": avg_prod_score,
                        "yearly_productivity_level": yearly_level,
                        "dominant_phase_percentage": dominant_phase_pct
                    }

                    yield "," + json.dumps(summary)

                yield "]"
            except Exception as e:
                yield json.dumps({"error": str(e)})

        return StreamingResponse(result_generator(), media_type="application/json")

    return endpoint

# ---- Register all 3 crops ----
app.post("/predict_paddy")(make_predict_endpoint("paddy"))
app.post("/predict_tea")(make_predict_endpoint("tea"))
app.post("/predict_coconut")(make_predict_endpoint("coconut"))
