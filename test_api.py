
import json
import pandas as pd
import joblib
import urllib.request
import urllib.error

# Load the saved model's expected feature names
saved = joblib.load("genetic_risk_model.joblib")
feature_columns = saved["feature_columns"]

# Get one sample row from the dataset
df = pd.read_csv("train.csv")
sample = df.drop(
    columns=["Disease_Risk", "Gene_ID"],
    errors="ignore"
).iloc[0]

# Build a JSON-compatible input
payload = {}

for column in feature_columns:
    value = sample[column]

    if pd.isna(value):
        payload[column] = None
    elif hasattr(value, "item"):
        payload[column] = value.item()
    else:
        payload[column] = value

print("Sending input:", payload)

request = urllib.request.Request(
    "http://127.0.0.1:5000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request) as response:
        print("HTTP status:", response.status)
        print("API response:", response.read().decode("utf-8"))

except urllib.error.HTTPError as error:
    print("HTTP error:", error.code)
    print(error.read().decode("utf-8"))

except urllib.error.URLError as error:
    print("Connection error:", error.reason)
