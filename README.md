# Sensor-Based Mastitis Detection in Dairy Cows

ML Zoomcamp midterm project by Azubuike Ochiagu.

## Problem and intended use

This project investigates whether udder sensor measurements and temperature
can classify a cow's recorded mastitis status.

The intended application is a prototype monitoring service that could flag
records for further examination. It detects recorded status; it does not
demonstrate prediction of future disease onset.

The service accepts nine numerical measurements and returns a model
probability and a classification at threshold 0.5.

## Dataset and attribution

Source: K, Ankitha; D H, Manjaiah; M, Kartik (2020).
“Data for: Clinical Mastitis in Cows based on Udder Parameter using
Internet of Things (IoT)”, Mendeley Data, Version 2.

- Dataset: https://doi.org/10.17632/kbvcdw5b4m.2
- Licence: Creative Commons Attribution 4.0
- Licence terms: https://creativecommons.org/licenses/by/4.0/
- Included file: `data/clinical_mastitis_cows.csv`

The CSV was extracted unchanged from the original archive. Images from the
archive are not used. Including the CSV avoids requiring reviewers to
download the approximately 1.6 GB archive.

Inspection of the CSV found:

- 6,600 records and 18 columns.
- 1,100 distinct cow IDs, each with six records.
- 3,961 normal records and 2,639 mastitis records.
- No missing values or exact duplicate rows.
- No repeated cow/day combinations.

These counts describe the downloaded file. Published descriptions of the
dataset differ in their reported animal and repeat counts.

Column names are standardised in memory. The original CSV is preserved.

## Model inputs

| Input | Dataset description |
|---|---|
| `iufl`, `eufl` | Front-left udder inhale/exhale sensor limits |
| `iufr`, `eufr` | Front-right udder inhale/exhale sensor limits |
| `iurl`, `eurl` | Rear-left udder inhale/exhale sensor limits |
| `iurr`, `eurr` | Rear-right udder inhale/exhale sensor limits |
| `temperature` | Recorded temperature value |

Inputs must use the original dataset's measurement scale. Sensor units and
the interpretation of temperature require clarification. Temperature values
range from 24 to 58 and are retained as recorded.

Cow ID, day, breed, months after giving birth, previous mastitis status,
hardness, pain, and milk visibility are excluded from the deployed model.

## Exploration and model selection

`notebook.ipynb` contains:

- Data quality checks and numerical summaries.
- Target distribution and sensor distributions by class.
- Correlation analysis.
- Comparison of a dummy baseline, logistic regression, random forest,
  and XGBoost.
- Random forest parameter tuning using three-fold GroupKFold.
- Validation permutation importance and threshold comparison.
- Final held-out test evaluation.

Cows are separated between training, validation, and test sets:

| Split | Records | Cow IDs |
|---|---:|---:|
| Training | 3,960 | 660 |
| Validation | 1,320 | 220 |
| Test | 1,320 | 220 |

GroupShuffleSplit uses random_state=42. Cow IDs do not overlap between
splits. Cross-validation also keeps cows separate.

The selected random forest uses:

- n_estimators=200
- max_depth=None
- min_samples_leaf=1
- random_state=42

The depth search tested [5, 10, None], and the minimum leaf-size search
tested [1, 5, 10], using cross-validation ROC AUC.

Threshold 0.5 gave the highest validation F1 among
[0.2, 0.3, 0.4, 0.5, 0.6].

After selection, the model was refitted on the combined training and
validation sets: 5,280 records from 880 cows.

## Final test results

The test set was reserved until model and threshold choices were fixed.

| Metric | Score |
|---|---:|
| ROC AUC | 0.9985 |
| Average precision | 0.9978 |
| Accuracy | 0.9848 |
| Precision | 0.9772 |
| Recall | 0.9876 |
| F1 | 0.9824 |
| Specificity | 0.9828 |

Confusion matrix: TN=743, FP=13, FN=7, TP=557.

Metrics are calculated per record, not per unique cow. Repeated records
within a cow are not independent observations.

## Limitations

This is an educational prototype, not a validated veterinary diagnostic tool.

- Evaluation covers held-out cow IDs from the same dataset.
- Generalisation to other farms, breeds, sensors, or collection conditions
  has not been established.
- Measurement units, temperature interpretation, and label-assignment
  procedures need further clarification.
- Model probabilities have not been calibrated as clinical risk estimates.
- Correlated sensor inputs complicate individual feature-importance
  interpretation.
- The threshold balances validation precision and recall; operational use
  would require agreed costs for missed cases and false alerts.

## Environment setup

The project was developed using Python 3.14.2.

From the repository root on Linux, macOS, or GitHub Codespaces:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the notebook

```bash
jupyter notebook notebook.ipynb
```

Select the project virtual environment as the notebook kernel.

The CSV is already included. For offline execution, replace the notebook's
download cell with:

```python
from pathlib import Path

data_path = Path("data/clinical_mastitis_cows.csv")
df = pd.read_csv(data_path)
```

## Train the selected model

```bash
python train.py
```

This recreates the development/test split, trains on development cows only,
and writes:

- `model.bin`: model, feature order, and threshold.
- `model_metadata.json`: parameters, record counts, library version,
  and dataset/model checksums.

Only load pickle artifacts from trusted sources.

## Run the API

```bash
python -m uvicorn predict:app --host 0.0.0.0 --port 9696
```

Endpoints:

- GET `/health`: service status and model checksum.
- POST `/predict`: probability and classification.
- GET `/docs`: interactive API documentation.

Example request using the original dataset's numerical scale:

```bash
curl -s http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "iufl": 150,
    "eufl": 180,
    "iufr": 150,
    "eufr": 180,
    "iurl": 150,
    "eurl": 181,
    "iurr": 150,
    "eurr": 181,
    "temperature": 43
  }'
```

The response contains `mastitis_probability`, `mastitis`, and `threshold`.

## Run with Docker

Build:

```bash
docker build -t mastitis-detection:midterm .
```

Stop any existing service using port 9696, then run:

```bash
docker run --rm -p 9696:9696 mastitis-detection:midterm
```

## Verify deployment

With the API or container running, execute in another terminal with the
project environment activated:

```bash
python test_api.py
```

The client checks:

- Service health and model checksum.
- Probability range and boolean classification.
- Agreement between API predictions and the saved model.
- Correct application of the saved threshold.

This checks deployment consistency, not predictive accuracy.

Local API and Docker prediction checks passed during development.
Cloud deployment is not included.

## Deployment evidence

The screenshot below shows the prediction checks passing and the Docker
container serving the API on port 9696.

![Docker deployment and prediction checks](deployment-test.png)