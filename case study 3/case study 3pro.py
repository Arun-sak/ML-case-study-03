# ============================================================
# CASE STUDY 3: HEART DISEASE CLASSIFICATION
# Algorithm: Random Forest Classifier
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

file_path = r"C:\Users\dkrup\OneDrive\ドキュメント\case study 3\heart_disease_uci.csv"

df = pd.read_csv(file_path)

print("\n========== HEART DISEASE CLASSIFICATION ==========")
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst five records:")
print(df.head())

# ------------------------------------------------------------
# 2. Identify target column
# ------------------------------------------------------------

column_names = {
    column.lower().strip(): column
    for column in df.columns
}

possible_targets = [
    "num",
    "target",
    "condition",
    "heartdisease",
    "heart disease",
    "output"
]

target_col = None

for target_name in possible_targets:
    if target_name in column_names:
        target_col = column_names[target_name]
        break

if target_col is None:
    raise ValueError(
        "Target column not found. Check the printed dataset columns."
    )

print("\nSelected target column:", target_col)

# ------------------------------------------------------------
# 3. Clean target
# ------------------------------------------------------------

df = df.dropna(subset=[target_col]).copy()

y = df[target_col]

# Convert UCI num values:
# 0 = no disease
# 1, 2, 3, 4 = disease

if target_col.lower().strip() == "num":

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid_target = y.notna()

    df = df.loc[valid_target].copy()
    y = y.loc[valid_target]

    y = (y > 0).astype(int)

else:

    if y.dtype == "object":

        y = y.astype(str).str.strip().str.lower()

        target_mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "presence": 1,
            "absence": 0,
            "disease": 1,
            "no disease": 0
        }

        if set(y.unique()).issubset(target_mapping.keys()):
            y = y.map(target_mapping)

if y.nunique() < 2:
    raise ValueError(
        "The target must contain at least two different classes."
    )

# ------------------------------------------------------------
# 4. Prepare feature data
# ------------------------------------------------------------

X = df.drop(columns=[target_col])

# Remove identifier columns if present
identifier_columns = [
    column for column in X.columns
    if column.lower().strip() in [
        "id",
        "patient_id",
        "unnamed: 0"
    ]
]

X = X.drop(
    columns=identifier_columns,
    errors="ignore"
)

# Convert numeric-looking object columns into numeric values
for column in X.columns:

    if X[column].dtype == "object":

        converted = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        if converted.notna().sum() >= (
            0.8 * X[column].notna().sum()
        ):
            X[column] = converted

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", numeric_columns)
print("\nCategorical features:", categorical_columns)

# ------------------------------------------------------------
# 5. Preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])

# ------------------------------------------------------------
# 6. Create Random Forest model
# ------------------------------------------------------------

model = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])

# ------------------------------------------------------------
# 7. Split dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 8. Train model
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# ------------------------------------------------------------
# 9. Predict test data
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 10. Evaluate model
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ------------------------------------------------------------
# 11. Feature importance
# ------------------------------------------------------------

trained_preprocessor = model.named_steps["preprocessing"]

trained_classifier = model.named_steps["classifier"]

feature_names = trained_preprocessor.get_feature_names_out()

importance_values = trained_classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")

print(
    importance_df.head(10).to_string(index=False)
)

# ------------------------------------------------------------
# 12. Example prediction from test data
# ------------------------------------------------------------

sample = X_test.iloc[[0]]

prediction = model.predict(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")

if prediction == 1:
    print("Predicted class: Heart disease")
else:
    print("Predicted class: No heart disease")

print("\nActual class:", y_test.iloc[0])
