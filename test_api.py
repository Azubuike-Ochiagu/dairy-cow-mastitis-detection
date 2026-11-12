import json
import pickle
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_URL = "http://localhost:9696"

# Check the health endpoint.
with urlopen(f"{BASE_URL}/health", timeout=10) as response:
    health = json.load(response)

metadata = json.loads(
    (ROOT / "model_metadata.json").read_text()
)

assert health["status"] == "ok"
assert health["model_sha256"] == metadata["model_sha256"]
print("Health check passed.")

# Load the saved model for comparison.
with (ROOT / "model.bin").open("rb") as source:
    artifact = pickle.load(source)

features = artifact["features"]
model = artifact["model"]

df = pd.read_csv(ROOT / "data" / "clinical_mastitis_cows.csv")
df.columns = df.columns.str.strip().str.lower()

# Check one example from each recorded class.
for label in [0, 1]:
    row = df.loc[df["class1"] == label, features].iloc[[0]]
    payload = {
        feature: float(row.iloc[0][feature])
        for feature in features
    }

    request = Request(
        f"{BASE_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        result = json.load(response)

    positive_index = list(model.classes_).index(1)
    expected = float(model.predict_proba(row)[0, positive_index])

    assert 0 <= result["mastitis_probability"] <= 1
    assert isinstance(result["mastitis"], bool)
    assert abs(result["mastitis_probability"] - expected) < 1e-12
    assert result["threshold"] == artifact["threshold"]
    assert result["mastitis"] == (
        expected >= artifact["threshold"]
    )

    print(f"\nExample with recorded class {label}:")
    print(json.dumps(result, indent=2))

print("\nPassed: API predictions match the saved model.")