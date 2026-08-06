import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional


# ============================================================
# STEP 1: Load locations.json (ONE TIME, when the app starts)
# We do NOT modify this file. We just read it into memory.
# ============================================================
with open("locations.json", "r", encoding="utf-8") as file:
    LOCATION_DATA_RAW = json.load(file)


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


LOCATION_DATA = build_lowercase_locations(LOCATION_DATA_RAW)


# ============================================================
# STEP 3: Load the trained ML model
# ============================================================
model = joblib.load("house_price_prediction_lrmodel_.pkl")


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
    return sorted(LOCATION_DATA.keys())


@app.get("/cities/{state}", response_model=list[str])
def get_cities(state: str):
    state = state.strip().lower()
    if state not in LOCATION_DATA:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    return sorted(LOCATION_DATA[state].keys())


@app.get("/localities/{state}/{city}", response_model=list[str])
def get_localities(state: str, city: str):
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

    log_prediction = model.predict(input_data)[0]

    # Model was trained on np.log1p(price), so we reverse it with np.expm1()
    actual_price = np.expm1(log_prediction)

    return PredictionResponse(predicted_price_inr=round(float(actual_price)))