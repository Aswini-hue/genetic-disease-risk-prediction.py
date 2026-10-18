
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("train.csv")
df = df.dropna(subset=["Disease_Risk"]).copy()

# Prepare features and target
X = df[["Gene_Expression", "Mutation"]]
y = df["Disease_Risk"].astype(str)

# Create and train model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

# Save model using local scikit-learn version
joblib.dump({
    "model": model,
    "input_columns": ["Gene_Expression", "Mutation"]
}, "genetic_risk_api_model.joblib")

print("Model trained and saved successfully!")
print("Classes:", model.classes_)
