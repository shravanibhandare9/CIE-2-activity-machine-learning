# Machine Learning (CIE-2) Activities

This repository contains practical coursework activities for Continuous Internal Evaluation 2 (CIE-2).

---

## Repository Structure

```
├── activity_no1/
│   ├── CIE2_Activity1_Dataset_Preparation.ipynb
│   ├── student_performance_dataset.csv
│   └── student_performance_prepared.csv
├── activity_no2/
│   ├── House-Price-Prediction.ipynb
│   └── House Price Prediction Dataset.csv
├── requirements.txt
└── README.md
```

---

## Activities Overview

### Activity 1: Dataset Preparation & Preprocessing
- **Objective:** Inspect tabular data, identify and handle data-quality issues (missing values, whitespace discrepancies, duplicates), perform exploratory data analysis (EDA), and prepare a cleaned, encoded, and scaled dataset.
- **Dataset:** Student Performance Dataset (`student_performance_dataset.csv`)
- **Key Preprocessing Steps:**
  - Categorical whitespace stripping
  - Numerical missing value imputation using column median
  - Categorical missing value imputation using column mode
  - Duplicate detection and removal
  - Exploratory visualizations (Distributions, Count plots, Box plots)
  - One-Hot Encoding for categorical features
  - Feature scaling via `StandardScaler`
  - Export of clean dataset (`student_performance_prepared.csv`)

### Activity 2: Simple House Price Prediction
- **Objective:** Implement Simple Linear Regression to predict continuous house prices based on property living area (`Area`).
- **Dataset:** House Price Prediction Dataset (`House Price Prediction Dataset.csv`)
- **Workflow:**
  - Feature selection: Input feature (`Area`), Target (`Price`)
  - Train-test split (80% training, 20% testing)
  - Model fitting with scikit-learn `LinearRegression`
  - Regression equation extraction: Price = Intercept + Slope * Area
  - Performance evaluation using MAE, MSE, RMSE, and R² score
  - Visualizations: Area vs. Price scatter plot and Actual vs. Predicted comparison plot
  - Prediction on new unseen property dimensions

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shravanibhandare9/CIE-2-activity-machine-learning.git
   cd CIE-2-activity-machine-learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
