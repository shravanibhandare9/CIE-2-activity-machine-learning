"""
=============================================================================
CIE-2 Machine Learning Activity: Dataset Preparation, EDA & Preprocessing
=============================================================================
Course: Machine Learning Lab / Continuous Internal Evaluation 2 (CIE-2)
Topic : End-to-End Data Pipeline on Student Academic Performance Dataset
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_pipeline():
    print("=" * 75)
    print(">>> STEP 1: DATASET INGESTION & STRUCTURAL INSPECTION")
    print("=" * 75)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(current_dir, "data", "student_performance_dataset.csv")
    prepared_path = os.path.join(current_dir, "data", "student_performance_prepared.csv")

    df = pd.read_csv(raw_path)
    print(f"Dataset successfully loaded from: {raw_path}")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Data Types ---")
    print(df.dtypes)

    print("\n" + "=" * 75)
    print(">>> STEP 2: MISSING VALUE & DUPLICATE AUDIT")
    print("=" * 75)
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_percent})
    print(missing_table[missing_table["Missing Count"] > 0])

    duplicate_count = df.duplicated().sum()
    print(f"\nExact Duplicate Rows Detected: {duplicate_count}")

    print("\n" + "=" * 75)
    print(">>> STEP 3: STATISTICAL PROFILING & EDA SUMMARY")
    print("=" * 75)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")

    print("\nNumerical Statistics:")
    print(df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]])

    print("\nCategorical Distributions:")
    for col in ["Gender", "Parental_Education", "Extracurricular_Activities", "Internet_Access"]:
        print(f"\nCounts for {col}:")
        print(df[col].value_counts(dropna=False))

    print("\n" + "=" * 75)
    print(">>> STEP 4: DATA CLEANING & PREPROCESSING")
    print("=" * 75)
    data = df.copy()

    # 4.1 Normalize strings & strip extra whitespaces
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()

    # 4.2 Remove duplicate rows
    initial_len = len(data)
    data = data.drop_duplicates().copy()
    print(f"Deduplication: Removed {initial_len - len(data)} duplicate rows. New shape: {data.shape}")

    # 4.3 Missing value imputation (Median for numeric, Mode for categorical)
    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].isna().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' using Median ({median_val:.2f})")

    for col in data.select_dtypes(include=["object"]).columns:
        if data[col].isna().any():
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
            print(f"Imputed missing values in '{col}' using Mode ('{mode_val}')")

    # 4.4 Outlier Handling via IQR Capping (Winsorization) on Study_Hours_Per_Week
    Q1 = data["Study_Hours_Per_Week"].quantile(0.25)
    Q3 = data["Study_Hours_Per_Week"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = ((data["Study_Hours_Per_Week"] < lower_bound) | (data["Study_Hours_Per_Week"] > upper_bound)).sum()
    print(f"Outlier Capping on 'Study_Hours_Per_Week': IQR={IQR:.2f}, Bounds=[{lower_bound:.2f}, {upper_bound:.2f}]. Detected: {outliers_count} outliers.")
    data["Study_Hours_Per_Week"] = data["Study_Hours_Per_Week"].clip(lower=lower_bound, upper=upper_bound)

    # 4.5 Drop identifier column
    if "Student_ID" in data.columns:
        data = data.drop(columns=["Student_ID"])

    # 4.6 Categorical Encoding (One-Hot with drop_first=True)
    cat_to_encode = [c for c in ["Gender", "Parental_Education", "Extracurricular_Activities", "Internet_Access"] if c in data.columns]
    encoded_df = pd.get_dummies(data, columns=cat_to_encode, drop_first=True, dtype=int)
    print(f"One-Hot Encoding completed. Shape before: {data.shape}, Shape after: {encoded_df.shape}")

    # 4.7 Feature Scaling using StandardScaler
    features_to_scale = [
        "Age", "Study_Hours_Per_Week", "Attendance_Rate", 
        "Tutoring_Sessions", "Sleep_Hours_Per_Night", "Previous_Exam_Score"
    ]
    scaler = StandardScaler()
    scaled_df = encoded_df.copy()
    scaled_df[features_to_scale] = scaler.fit_transform(scaled_df[features_to_scale])
    print(f"StandardScaler applied to features: {features_to_scale}")

    # 4.8 Export prepared dataset
    scaled_df.to_csv(prepared_path, index=False)
    print(f"Successfully saved clean prepared dataset to: {prepared_path}")

    print("\n" + "=" * 75)
    print(">>> STEP 5: TRAIN-TEST SPLIT & BASELINE MODEL EVALUATION")
    print("=" * 75)
    X = scaled_df.drop(columns=["Final_Exam_Score"])
    y = scaled_df["Final_Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training Features Shape: {X_train.shape}, Test Features Shape: {X_test.shape}")
    print(f"Training Target Shape  : {y_train.shape}, Test Target Shape  : {y_test.shape}")

    # Train Linear Regression Baseline
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nBaseline Model Performance (Linear Regression):")
    print(f"  * Mean Absolute Error (MAE) : {mae:.3f}")
    print(f"  * Root Mean Squared Error   : {rmse:.3f}")
    print(f"  * R-squared (R2 Score)      : {r2:.4f} ({r2*100:.2f}% variance explained)")

    print("\n" + "=" * 75)
    print(">>> CIE-2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
