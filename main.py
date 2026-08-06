import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
from pathlib import Path
import logging


# ============================================================
# STEP 1: Load locations.json (ONE TIME, when the app starts)
# We do NOT modify this file. We just read it into memory.
# ============================================================
# Lazy placeholders — actual loading happens on startup to allow clear startup errors
LOCATION_DATA_RAW = None
LOCATION_DATA = {}


# ============================================================
# STEP 2: Build a LOWERCASE copy of the location data.
# This is only used for validation. The original file/dict
# above (LOCATION_DATA_RAW) is never changed.
#
# Result looks like:
# {
#   "odisha": {
#       "bhubaneswar": {"patia", "nayapalli", "khandagiri"}
#   }
# }
#
# Works whether your JSON stores localities as a list:
#   "Bhubaneswar": ["Patia", "Nayapalli"]
# or as a dict:
#   "Bhubaneswar": {"Patia": {}, "Nayapalli": {}}
# ============================================================
def build_lowercase_locations(raw_data: dict) -> dict:
    lower_data = {}
    for state, cities in raw_data.items():
        lower_data[state.lower()] = {}
        for city, localities in cities.items():
            if isinstance(localities, dict):
                locality_names = localities.keys()
            else:
                locality_names = localities  # it's a list
            lower_data[state.lower()][city.lower()] = {
                loc.lower() for loc in locality_names
            }
    return lower_data


# LOCATION_DATA will be populated on startup by loading locations.json
# LOCATION_DATA = build_lowercase_locations(LOCATION_DATA_RAW)


# ============================================================
# STEP 3: Load the trained ML model
# ============================================================
# Model is loaded on startup to provide clearer errors if file is missing/corrupt
model = None


# ============================================================
# STEP 4: Create the FastAPI app
# ============================================================
app = FastAPI(title="House Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Setup logging
logger = logging.getLogger("house-price-api")
logging.basicConfig(level=logging.INFO)

# Paths (resolved relative to this file) — makes app robust to working directory
BASE_DIR = Path(__file__).parent
LOCATIONS_PATH = BASE_DIR / "locations.json"
MODEL_PATH = BASE_DIR / "house_price_prediction_lrmodel_.pkl"

# Startup state
startup_error: Optional[str] = None


def load_location_data(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load locations.json from {path}: {e}") from e
    return build_lowercase_locations(raw)


def load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e


@app.on_event("startup")
def startup_event():
    """Load locations.json and ML model on startup with clear logging/errors."""
    global LOCATION_DATA_RAW, LOCATION_DATA, model, startup_error
    try:
        logger.info("Loading locations from %s", LOCATIONS_PATH)
        LOCATION_DATA_RAW = json.loads(LOCATIONS_PATH.read_text(encoding="utf-8"))
        LOCATION_DATA = build_lowercase_locations(LOCATION_DATA_RAW)
        logger.info("Loaded locations: %d states", len(LOCATION_DATA))
    except Exception as e:
        startup_error = str(e)
        logger.exception("Error loading locations.json: %s", e)

    try:
        logger.info("Loading model from %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        startup_error = (startup_error + "; " if startup_error else "") + str(e)
        logger.exception("Error loading model: %s", e)


@app.get("/health")
def health():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Startup error: {startup_error}")
    return {"status": "ok", "model_loaded": model is not None, "states": len(LOCATION_DATA)}


# List of fields that must always be lowercased (all text/category fields)
TEXT_FIELDS = (
    "State", "City", "Locality", "Property_Type", "Facing",
    "Furnishing_Status", "Parking", "Lift", "Gym", "Swimming_Pool",
    "Garden", "Security", "Condition",
)


# ============================================================
# STEP 5: Pydantic model for the incoming request
# ============================================================
class HouseData(BaseModel):
    State: str = Field(..., min_length=2, max_length=50)
    City: str = Field(..., min_length=2, max_length=50)
    Locality: Optional[str] = Field(default=None, min_length=2, max_length=100)
    PIN_Code: Optional[int] = Field(default=None, ge=100000, le=999999)
    Property_Type: Literal["apartment", "villa", "independent house"]
    BHK: int = Field(..., ge=1, le=10)
    Bathrooms: int = Field(..., ge=1, le=10)
    Balcony: int = Field(..., ge=0, le=10)
    Facing: Literal[
        "north", "south", "east", "west",
        "north-east", "north-west", "south-east", "south-west",
    ]
    Floor: int = Field(..., ge=0)
    Builtup_Area_sqft: float = Field(..., gt=0)
    Plot_Area_sqft: Optional[float] = Field(default=None, ge=0)
    Property_Age: int = Field(..., ge=0, le=150)
    Furnishing_Status: Literal["unfurnished", "semi furnished", "furnished"]
    Parking: Literal["yes", "no"]
    Lift: Literal["yes", "no"]
    Gym: Literal["yes", "no"]
    Swimming_Pool: Literal["yes", "no"]
    Garden: Literal["yes", "no"]
    Security: Literal["yes", "no"]
    School_Distance_km: float = Field(..., ge=0)
    Hospital_Distance_km: float = Field(..., ge=0)
    Metro_Distance_km: float = Field(..., ge=0)
    Mall_Distance_km: float = Field(..., ge=0)
    Condition: Literal["poor", "fair", "good", "excellent", "luxury"]
    Maintenance_Fee_INR: float = Field(..., ge=0)
    Property_Tax_INR_Per_Year: float = Field(..., ge=0)
    Crime_Rate_Index: float = Field(..., ge=0, le=100)
    AQI: float = Field(..., ge=0, le=500)

    # --- Runs FIRST, before any other check ---
    # Converts "ODISHA" / "Odisha" / "odisha" -> "odisha"
    @field_validator(*TEXT_FIELDS, mode="before")
    @classmethod
    def lowercase_text_fields(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    # --- Runs AFTER all fields above are validated ---
    # Checks that State -> City -> Locality actually match locations.json
    @model_validator(mode="after")
    def check_location_is_valid(self):
        state = self.State
        city = self.City
        locality = self.Locality

        if state not in LOCATION_DATA:
            raise ValueError(f"Unknown State: '{state}'")

        if city not in LOCATION_DATA[state]:
            raise ValueError(f"City '{city}' does not belong to State '{state}'")

        if locality is not None and locality not in LOCATION_DATA[state][city]:
            raise ValueError(
                f"Locality '{locality}' does not belong to City '{city}', State '{state}'"
            )

        return self


class PredictionResponse(BaseModel):
    predicted_price_inr: float = Field(..., ge=0)


# ============================================================
# STEP 6: Routes
# ============================================================
@app.get("/")
def greet():
    return {"message": "Hello, welcome to the House Price Prediction API!"}


# --- Dropdown helper endpoints for the frontend ---

@app.get("/states", response_model=list[str])
def get_states():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")
    if not LOCATION_DATA:
        raise HTTPException(status_code=503, detail="Location data not loaded.")
    return sorted(LOCATION_DATA.keys())


@app.get("/cities/{state}", response_model=list[str])
def get_cities(state: str):
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")
    if not LOCATION_DATA:
        raise HTTPException(status_code=503, detail="Location data not loaded.")
    state = state.strip().lower()
    if state not in LOCATION_DATA:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    return sorted(LOCATION_DATA[state].keys())


@app.get("/localities/{state}/{city}", response_model=list[str])
def get_localities(state: str, city: str):
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")
    if not LOCATION_DATA:
        raise HTTPException(status_code=503, detail="Location data not loaded.")
    state = state.strip().lower()
    city = city.strip().lower()
    if state not in LOCATION_DATA:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    if city not in LOCATION_DATA[state]:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found under '{state}'.")
    return sorted(LOCATION_DATA[state][city])


# --- Main prediction endpoint ---

@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseData):
    # Ensure app started up correctly
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")
    if model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    # At this point:
    # - every text field is already lowercase
    # - State/City/Locality are already confirmed valid against locations.json
    input_data = pd.DataFrame([{
        "State": data.State,
        "City": data.City,
        "Locality": data.Locality,
        "PIN_Code": data.PIN_Code,
        "Property_Type": data.Property_Type,
        "BHK": data.BHK,
        "Bathrooms": data.Bathrooms,
        "Balcony": data.Balcony,
        "Facing": data.Facing,
        "Floor": data.Floor,
        "Builtup_Area_sqft": data.Builtup_Area_sqft,
        "Plot_Area_sqft": data.Plot_Area_sqft,
        "Property_Age": data.Property_Age,
        "Furnishing_Status": data.Furnishing_Status,
        "Parking": data.Parking,
        "Lift": data.Lift,
        "Gym": data.Gym,
        "Swimming_Pool": data.Swimming_Pool,
        "Garden": data.Garden,
        "Security": data.Security,
        "School_Distance_km": data.School_Distance_km,
        "Hospital_Distance_km": data.Hospital_Distance_km,
        "Metro_Distance_km": data.Metro_Distance_km,
        "Mall_Distance_km": data.Mall_Distance_km,
        "Condition": data.Condition,
        "Maintenance_Fee_INR": data.Maintenance_Fee_INR,
        "Property_Tax_INR_Per_Year": data.Property_Tax_INR_Per_Year,
        "Crime_Rate_Index": data.Crime_Rate_Index,
        "AQI": data.AQI,
    }])

    try:
        log_prediction = model.predict(input_data)[0]
        # Model was trained on np.log1p(price), so we reverse it with np.expm1()
        actual_price = np.expm1(log_prediction)
        predicted_price = float(round(actual_price, 2))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(predicted_price_inr=predicted_price)
