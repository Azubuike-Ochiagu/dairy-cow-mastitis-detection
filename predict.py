"""Serve the mastitis classifier through FastAPI."""

import hashlib
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.bin"

with MODEL_PATH.open("rb") as source:
    artifact = pickle.load(source)

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]

model_checksum = hashlib.sha256(
    MODEL_PATH.read_bytes()
).hexdigest()

app = FastAPI(
    title="Dairy Cow Mastitis Detection",
    description=(
        "Educational prototype for classifying recorded mastitis "
        "status. Inputs must use the original dataset's sensor scale."
    ),
)


class SensorMeasurements(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    iufl: float = Field(ge=0)
    eufl: float = Field(ge=0)
    iufr: float = Field(ge=0)
    eufr: float = Field(ge=0)
    iurl: float = Field(ge=0)
    eurl: float = Field(ge=0)
    iurr: float = Field(ge=0)
    eurr: float = Field(ge=0)
    temperature: float = Field(ge=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_sha256": model_checksum,
    }


@app.post("/predict")
def predict(measurements: SensorMeasurements):
    row = pd.DataFrame(
        [measurements.model_dump()],
        columns=features,
    )

    positive_class_index = list(model.classes_).index(1)

    probability = float(
        model.predict_proba(row)[0, positive_class_index]
    )

    return {
        "mastitis_probability": probability,
        "mastitis": bool(probability >= threshold),
        "threshold": threshold,
    }