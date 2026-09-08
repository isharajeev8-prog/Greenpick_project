from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. ML Models Loading (Pickle Files) ---
crop_model = None
fertilizer_model = None

try:
    with open("crop_recommendation_model - Copy.pkl", "rb") as f1:
        crop_model = pickle.load(f1)
    print("ML Models loaded successfully!")
except Exception as e:
    print(f"Error loading pickle files: {e}")



PROFIT_DATABASE = {
    "rice": {"cost_per_acre": 15000, "revenue_per_acre": 45000},
    "maize": {"cost_per_acre": 12000, "revenue_per_acre": 35000},
    "coconut": {"cost_per_acre": 20000, "revenue_per_acre": 60000},
    "banana": {"cost_per_acre": 25000, "revenue_per_acre": 85000},
    "kidneybeans": {"cost_per_acre": 30000, "revenue_per_acre": 40000},
    "chickpeas": {"cost_per_acre": 30000, "revenue_per_acre": 17000},
    "wheat": {"cost_per_acre": 50000, "revenue_per_acre": 100000},
    "sugarcane": {"cost_per_acre": 30000, "revenue_per_acre": 45000},
    "coffee": {"cost_per_acre": 38000, "revenue_per_acre": 90000}
}

# --- Pydantic Models ---

class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class FertilizerInput(BaseModel):
    temperature: float
    moisture: float
    soil_type: str
    crop_type: str


class ProfitInput(BaseModel):
    crop_name: str
    land_area: float


# --- #page1: Crop Prediction Route ---
@app.post("/predict")
def predict_crop(input_data: CropInput):
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded on server")
    try:
        features = np.array([[
            input_data.nitrogen, input_data.phosphorus, input_data.potassium,
            input_data.temperature, input_data.humidity, input_data.ph, input_data.rainfall
        ]])
        prediction = crop_model.predict(features)
        return {"recommended_crop": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- #page2: Fertilizer Prediction Route ---
@app.post("/predict-fertilizer-db")
async def predict_fertilizer(data: FertilizerInput):
    try:

        soil = data.soil_type.strip().lower()
        crop = data.crop_type.strip().lower()
        temp = data.temperature
        moisture = data.moisture

        recommended_fertilizer = ""

        # --- 1. SANDY SOIL ---
        if soil == "sandy":
            if crop == "maize":
                if temp > 25 and moisture < 50:
                    recommended_fertilizer = "Urea (Nitrogen Rich)"
                else:
                    recommended_fertilizer = "NPK 20:20:20"
            elif crop == "groundnuts":
                if moisture < 40:
                    recommended_fertilizer = "Single Super Phosphate (SSP)"
                else:
                    recommended_fertilizer = "Gypsum"
            else:
                recommended_fertilizer = "Generic NPK for Sandy Soil"

        # --- 2. CLAYEY SOIL ---
        elif soil == "clayey":
            if crop == "rice":
                if temp > 28 and moisture > 60:
                    recommended_fertilizer = "DAP (Di-Ammonium Phosphate)"
                else:
                    recommended_fertilizer = "Ammonium Sulfate"
            else:
                recommended_fertilizer = "NPK 10:26:26"

        # --- 3. LOAMY SOIL ---
        elif soil == "loamy":
            if crop == "cotton":
                if temp > 22 and moisture > 50:
                    recommended_fertilizer = "MOP (Muriate of Potash)"
                else:
                    recommended_fertilizer = "Urea + DAP Mix"
            elif crop == "sugarcane":
                recommended_fertilizer = "NPK 19:19:19"
            else:
                recommended_fertilizer = "Organic Compost / Vermicompost"

        # --- 4. BLACK SOIL  ---
        elif soil == "black":
            if crop == "cotton":
                if temp > 30:
                    recommended_fertilizer = "Sulfate of Potash"
                else:
                    recommended_fertilizer = "DAP"
            else:
                recommended_fertilizer = "NPK Complex Fertilizer"

        # --- 5. RED SOIL  ---
        elif soil == "red":
            if moisture < 45:
                recommended_fertilizer = "Super Phosphate"
            else:
                recommended_fertilizer = "NPK 15:15:15"

        #(Default)
        else:
            recommended_fertilizer = "Balanced NPK (19:19:19) with Organic Manure"

        #
        return {
            "status": "success",
            "recommended_fertilizer": recommended_fertilizer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- #page3: Profit Estimation Route ---
@app.post("/estimate-profit")
def estimate_profit(input_data: ProfitInput):
    crop = input_data.crop_name.lower().strip()
    area = input_data.land_area

    if crop in PROFIT_DATABASE:
        crop_data = PROFIT_DATABASE[crop]
        total_cost = crop_data["cost_per_acre"] * area
        total_revenue = crop_data["revenue_per_acre"] * area
        net_profit = total_revenue - total_cost

        return {
            "crop": crop,
            "land_area_acres": area,
            "total_cost": round(total_cost, 2),
            "total_revenue": round(total_revenue, 2),
            "net_profit": round(net_profit, 2)
        }
    else:
        raise HTTPException(status_code=404, detail="Selected crop not found in database")