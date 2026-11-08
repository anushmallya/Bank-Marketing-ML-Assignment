# Bank Marketing Classification – Machine Learning Assignment

## Project Overview
This project implements and evaluates five machine learning classification algorithms to predict whether a customer will subscribe to a term deposit based on their demographic information and previous marketing campaign interactions.
The objective is to compare different classification models using multiple evaluation metrics and identify the model that performs best at predicting potential subscribers.

## Dataset Description
- **Dataset:** Bank Marketing
- **Source:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Total Records:** 45,211
- **Original Features:** 16 input features
- **Target Variable:** `y` (yes/no)

The dataset contains customer demographic details, account information, and marketing campaign interaction data.

The `duration` feature was excluded from model training because call duration is only known after a call has ended and would introduce target leakage.

The final modelling dataset contains:
- 15 input features
- 1 target variable (`y`)

The target variable was encoded as:
- `0` – No subscription
- `1` – Subscription

The dataset has an imbalanced target distribution, with fewer customers subscribing to term deposits.

## Project Workflow

1. **Data Collection:** Retrieved the Bank Marketing dataset from the UCI Machine Learning Repository.
2. **Exploratory Data Analysis:** Examined feature distributions, missing values, duplicate records, and target class distribution.
3. **Data Preprocessing:**
   - Handled missing numerical values using median imputation.
   - Handled missing categorical values using mode or an `unknown` category, depending on the feature.
   - Applied StandardScaler to numerical features.
   - Applied OneHotEncoder to categorical features.
   - Excluded the `duration` feature to prevent target leakage.
4. **Train-Test Split:** Split the dataset into 80% training data and 20% testing data using stratification.
5. **Model Training:** Implemented five classification algorithms.
6. **Model Evaluation:** Compared model performance using six evaluation metrics.
7. **Model Saving:** Saved the trained models and preprocessing pipeline for reuse.

## Machine Learning Models

The following classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

## Model Evaluation

All five models were evaluated on the same test dataset containing 9,043 records.

### Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8934 | 0.7718 | 0.6655 | 0.1786 | 0.2817 | 0.3073 |
| Decision Tree | 0.8335 | 0.6167 | 0.3059 | 0.3336 | 0.3192 | 0.2248 |
| KNN | 0.8878 | 0.7045 | 0.5558 | 0.2023 | 0.2966 | 0.2879 |
| Naive Bayes | 0.8444 | 0.7517 | 0.3681 | 0.4603 | 0.4091 | 0.3234 |
| Support Vector Machine | 0.8932 | 0.7682 | 0.6513 | 0.1871 | 0.2907 | 0.3101 |

### Model-wise Observations

**1. Logistic Regression**
- Achieved the highest accuracy of 89.34%.
- Recorded the highest AUC of 0.7718 and precision of 0.6655.
- However, its recall of 0.1786 indicates that it missed many actual subscribers.

**2. Decision Tree**
- Achieved a recall of 0.3336, higher than Logistic Regression, KNN, and SVM.
- Its accuracy and AUC were comparatively lower.
- The model identified more actual subscribers but also produced more false positives.

**3. K-Nearest Neighbors (KNN)**
- Achieved an accuracy of 88.78%.
- Its precision was 0.5558, but recall remained relatively low at 0.2023.
- It performed better in precision than Decision Tree and Naive Bayes but missed many actual subscribers.

**4. Gaussian Naive Bayes**
- Achieved the highest recall of 0.4603 and F1-score of 0.4091.
- Also recorded the highest MCC of 0.3234.
- It identified the largest number of actual subscribers among the five models, although it generated more false positives.

**5. Support Vector Machine**
- Achieved an accuracy of 89.32% and AUC of 0.7682.
- Its precision was 0.6513, but recall was relatively low at 0.1871.
- Its overall performance was close to Logistic Regression.

## Overall Model Comparison and Conclusion

Based on the evaluated baseline models, **Gaussian Naive Bayes performed best in terms of recall, F1-score, and Matthews Correlation Coefficient (MCC)**.

It identified more actual subscribers than the other models, making it a suitable choice when identifying potential subscribers is a priority.

However, Logistic Regression achieved the highest accuracy, AUC, and precision. Therefore, the choice of model depends on the business objective and the relative importance of false positives and false negatives.

The results also demonstrate why accuracy alone is not sufficient for evaluating models on an imbalanced dataset.

## Saved Artifacts

The repository contains the following saved artifacts:

- Five trained classification models in `.pkl` format.
- `preprocessor.pkl` containing the fitted preprocessing pipeline.
- `test_data.csv` containing the held-out test dataset with original input features and actual target labels.
- `model_evaluation_results.csv` containing the model evaluation results.

These artifacts can be reused for model evaluation without retraining the models.
