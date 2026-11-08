from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Subscription Prediction")
st.write(
    "Evaluate the performance of saved machine learning "
    "models using uploaded test data."
)

# --------------------------------------------------
# 2. Artifact paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Assignment" / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

FEATURE_COLUMNS = [
    "age", "balance", "day_of_week", "campaign", "pdays", "previous",
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]

# --------------------------------------------------
# 3. Upload test dataset
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# 4. Select model
# --------------------------------------------------

selected_model = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys())
)

# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write(f"Number of records: {len(test_data)}")
        st.dataframe(test_data.head())

        # Validate required columns
        required_columns = FEATURE_COLUMNS + ["y"]
        missing_columns = [
            col for col in required_columns
            if col not in test_data.columns
        ]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")

        elif st.button("Evaluate Model"):

            # Separate input features and actual target
            X_test = test_data[FEATURE_COLUMNS].copy()

            target = test_data["y"]

            # Normalize target values
            normalized_target = target.astype(str).str.strip().str.lower()

            # Convert yes/no and numeric labels into 0/1
            y_test = normalized_target.map({
                "no": 0,
                "yes": 1,
                "0": 0,
                "1": 1,
                "0.0": 0,
                "1.0": 1
            })

            # Validate target values
            if y_test.isna().any():
                st.error(
                    f"Unexpected target values: {target.unique().tolist()}"
                )
                st.stop()

            y_test = y_test.astype(int)

            if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
                st.error("Target column 'y' must contain only yes/no or 0/1.")
                st.stop()

            y_test = y_test.astype(int)

            # Load saved preprocessing pipeline and model
            preprocessor = joblib.load(PREPROCESSOR_PATH)

            model_path = MODEL_DIR / MODEL_FILES[selected_model]
            model = joblib.load(model_path)

            # Transform test features
            X_processed = preprocessor.transform(X_test)

            # Generate predictions
            y_pred = model.predict(X_processed)

            # Generate scores for AUC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_processed)[:, 1]
            else:
                y_score = model.decision_function(X_processed)

            # Calculate evaluation metrics
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_score),
                "Precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "F1-Score": f1_score(
                    y_test, y_pred, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

            # Display model results
            st.subheader(f"Evaluation Results: {selected_model}")

            col1, col2, col3 = st.columns(3)

            metric_items = list(metrics.items())

            for i, (name, value) in enumerate(metric_items):
                columns = [col1, col2, col3]
                with columns[i % 3]:
                    st.metric(name, f"{value:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(cm_df)

            # Classification report
            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                target_names=["No Subscription", "Subscription"],
                output_dict=True,
                zero_division=0
            )

            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as error:
        st.error(f"Unable to evaluate the uploaded dataset: {error}")

else:
    st.info("Please upload your test_data.csv file to begin evaluation.")