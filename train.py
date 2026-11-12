"""Train the selected mastitis classifier and save its artifact."""

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clinical_mastitis_cows.csv"
MODEL_PATH = ROOT / "model.bin"

FEATURES = [
    "iufl", "eufl",
    "iufr", "eufr",
    "iurl", "eurl",
    "iurr", "eurr",
    "temperature",
]

THRESHOLD = 0.5


def main():
    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    df = df.rename(columns={
        "class1": "mastitis",
        "previous_mastits_status": "previous_mastitis_status",
    })

    # Reproduce the notebook's development/test split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    development_idx, test_idx = next(
        splitter.split(df, groups=df["cow_id"])
    )

    development = df.iloc[development_idx]
    test = df.iloc[test_idx]

    assert set(development["cow_id"]).isdisjoint(
        set(test["cow_id"])
    )

    # Parameters selected before the final test evaluation.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
    )

    model.fit(
        development[FEATURES],
        development["mastitis"],
    )

    artifact = {
        "model": model,
        "features": FEATURES,
        "threshold": THRESHOLD,
    }

    with MODEL_PATH.open("wb") as output:
        pickle.dump(
            artifact,
            output,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metadata = {
        "model_type": "RandomForestClassifier",
        "parameters": model.get_params(),
        "features": FEATURES,
        "threshold": THRESHOLD,
        "training_records": len(development),
        "training_cows": development["cow_id"].nunique(),
        "held_out_records": len(test),
        "held_out_cows": test["cow_id"].nunique(),
        "random_state": 42,
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(
            DATA_PATH.read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(
            MODEL_PATH.read_bytes()
        ).hexdigest(),
    }

    (ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"Trained on {len(development)} records "
        f"from {development['cow_id'].nunique()} cows."
    )
    print(f"Saved model: {MODEL_PATH.name}")
    print("Saved metadata: model_metadata.json")


if __name__ == "__main__":
    main()