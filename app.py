from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

artifact = joblib.load("models/cardio_model.joblib")
model = artifact["model"]
features = artifact["features"]

class CardioInput(BaseModel):
    age: float
    gender: float
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: float
    gluc: float
    smoke: int
    alco: int
    active: int

@app.post("/predict")
def predict(data: CardioInput):
    age_years = int(data.age) / 365
    bmi = data.weight / (data.height/ 100)**2
    row = pd.DataFrame([[
        data.gender, data.ap_hi, data.ap_lo, data.cholesterol, data.gluc, data.smoke, data.alco, data.active, age_years, bmi
    ]], columns=features)

    proba = model.predict_proba(row)[0, 1]
    threshold = artifact["threshold"]
    pred = int(proba >= threshold)
    return {"prediction": pred, "probability": round(float(proba), 3)}