
import json
import joblib
import urllib.request
import urllib.error

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Genetic Data Analysis",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Data Compression and Pattern Visualization System")
st.write("Data Preprocessing, EDA, PCA, LDA and Disease Risk Prediction")


# ---------------- STEP 1: LOAD DATA ----------------
try:
    data = pd.read_csv("train.csv")
except FileNotFoundError:
    st.error("train.csv must be in the same folder as dashboard.py")
    st.stop()
except Exception as error:
    st.error(f"Could not read dataset: {error}")
    st.stop()

if data.empty:
    st.error("Your dataset is empty.")
    st.stop()

target = "Disease_Risk"

if target not in data.columns:
    st.error("Dataset must contain a Disease_Risk column.")
    st.stop()

st.header("1. Dataset Overview")
st.write("Original Dataset Shape:", data.shape)
st.metric(
    "Original Number of Features",
    data.drop(columns=["Gene_ID", "Disease_Risk"], errors="ignore").shape[1]
)
st.write("Column Names:", data.columns.tolist())
st.dataframe(data.head(10), use_container_width=True)


# ---------------- STEP 2: DATA PREPROCESSING ----------------
st.header("2. Data Preprocessing")

st.subheader("Missing Values Before Cleaning")
st.dataframe(data.isnull().sum().rename("Missing Values"))

clean_data = data.copy()

# Remove duplicate rows
duplicate_count = int(clean_data.duplicated().sum())
clean_data = clean_data.drop_duplicates().reset_index(drop=True)

st.write("Duplicate rows removed:", duplicate_count)

# Fill missing values
for col in clean_data.columns:
    if clean_data[col].isnull().any():
        if pd.api.types.is_numeric_dtype(clean_data[col]):
            median_value = clean_data[col].median()
            clean_data[col] = clean_data[col].fillna(
                median_value if pd.notna(median_value) else 0
            )
        else:
            modes = clean_data[col].mode(dropna=True)
            fill_value = (
                modes.iloc[0] if not modes.empty else "Unknown"
            )
            clean_data[col] = clean_data[col].fillna(fill_value)

st.subheader("Cleaned Dataset")
st.dataframe(clean_data, use_container_width=True)

st.subheader("Missing Values After Cleaning")
st.dataframe(clean_data.isnull().sum().rename("Missing Values"))

# STEP 3: PCA COMPONENTS
st.header("3. PCA Components")

st.write("Original Number of Features:", clean_data.drop(
    columns=["Gene_ID", "Disease_Risk"], errors="ignore"
).shape[1])

st.write("PCA Components Selected: 2")

st.subheader("Explained Variance")

# Actual PCA explained variance
pca_features = clean_data.drop(
    columns=["Gene_ID", "Disease_Risk"],
    errors="ignore"
).select_dtypes(include="number")

if pca_features.shape[1] >= 2 and len(pca_features) >= 2:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaled_features = StandardScaler().fit_transform(pca_features)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    variance = pca.explained_variance_ratio_

    st.write(f"PC1 Explained Variance: {variance[0] * 100:.2f}%")
    st.write(f"PC2 Explained Variance: {variance[1] * 100:.2f}%")
    st.write(f"Total Explained Variance: {variance.sum() * 100:.2f}%")
else:
    st.warning("PCA needs at least 2 numeric features and 2 rows.")


st.info(
    "Explained variance shows how much information "
    "each PCA component retains from the original data."
)

st.subheader("2D PCA Visualization")

if pca_features.shape[1] >= 2 and len(pca_features) >= 2:
    pca_df = pd.DataFrame(
        pca_result,
        columns=["PC1", "PC2"]
    )

    fig, ax = plt.subplots()
    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"]
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("2D PCA Visualization")
    ax.grid(True)

    st.pyplot(fig)
else:
    st.warning("Not enough numeric features for 2D PCA.")
# LDA Visualization
st.subheader("LDA Visualization")

lda_features = clean_data.drop(
    columns=["Gene_ID", "Disease_Risk"],
    errors="ignore"
).select_dtypes(include="number")

if "Disease_Risk" in clean_data.columns:
    lda_labels = clean_data["Disease_Risk"]

    if lda_features.shape[1] >= 1 and lda_labels.nunique() >= 2:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler

        lda_scaled = StandardScaler().fit_transform(lda_features)

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda_result = lda.fit_transform(lda_scaled, lda_labels)

        fig_lda, ax_lda = plt.subplots()
        ax_lda.scatter(
            lda_result[:, 0],
            [0] * len(lda_result),
            c=pd.factorize(lda_labels)[0]
        )
        ax_lda.set_xlabel("Linear Discriminant 1")
        ax_lda.set_title("LDA Visualization by Disease Risk")
        ax_lda.set_yticks([])

        st.pyplot(fig_lda)
    else:
        st.warning("LDA needs numeric features and at least 2 risk classes.")
else:
    st.warning("Disease_Risk column not found.")

st.header("PCA vs LDA Comparison")

comparison = pd.DataFrame({
    "Method": ["PCA", "LDA"],
    "Type": [
        "Unsupervised Learning",
        "Supervised Learning"
    ],
    "Main Purpose": [
        "Reduce dimensions while retaining variance",
        "Separate disease risk classes"
    ],
    "Uses Disease Labels": ["No", "Yes"]
})

st.dataframe(comparison, use_container_width=True)

# ---------------- STEP 4: FLASK API PREDICTION ----------------
st.header("4. Disease Risk Prediction")

st.write(
    "Enter the gene expression and mutation values "
    "to predict disease risk using the Flask API."
)

with st.form("prediction_form"):
    gene_expression = st.number_input(
        "Gene Expression",
        value=5.2,
        step=0.1,
        format="%.2f"
    )

    mutation = st.selectbox(
        "Mutation",
        options=[0, 1],
        format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)"
    )

    submitted = st.form_submit_button("Predict Disease Risk")

if submitted:
    payload = {
        "Gene_Expression": gene_expression,
        "Mutation": mutation
    }

    try:
        request = urllib.request.Request(
            "http://127.0.0.1:5000/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))

        if "predicted_disease_risk" in result:
            st.success(
                "Predicted Disease Risk: "
                + result["predicted_disease_risk"]
            )
            st.caption(result.get("note", ""))
        else:
            st.error(result.get("error", "Prediction failed."))

    except urllib.error.URLError:
        st.error(
            "Cannot connect to Flask API. "
            "Please start Flask using: python app.py"
        )
    except Exception as error:
        st.error(f"Prediction error: {error}")
        
# ===== Google Colab Analysis Results =====

st.header("5. Google Colab Analysis Results")

import os

project_folder = os.path.dirname(os.path.abspath(__file__))

# Load Colab CSV results

colab_files = {
    "PCA Visualization": "pca_visualization (2).csv",
    "Class-wise Risk Analysis": "class_kpis (2).csv",
    "PCA Explained Variance": "explained_variance (2).csv",
}

    

for section_name, file_name in colab_files.items():
    file_path = os.path.join(project_folder, file_name)

    st.subheader(section_name)

    if os.path.exists(file_path):
        result_df = pd.read_csv(file_path)
        st.dataframe(result_df, use_container_width=True)

        if section_name == "PCA Explained Variance":
            st.bar_chart(result_df)
    else:
        st.warning(f"{file_name} not found in project folder.")



