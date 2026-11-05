# 🎓 Machine Learning CIE-2 Activity

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Continuous Internal Evaluation 2 (CIE-2) Machine Learning Activity**  
> Complete lab work containing dataset preparation, exploratory data analysis, data preprocessing, and predictive model training using Python.

---

## 📁 Repository Structure

```text
CIE-2-activity-machine-learning/
├── README.md                                           # Comprehensive documentation
├── requirements.txt                                    # Python dependencies
├── .gitignore                                          # Git ignore file
├── run_pipeline.py                                     # Python script reproducing the full workflow
│
├── activity_no1/                                       # ACTIVITY NO. 1: Dataset Preparation & Preprocessing
│   ├── CIE2_Activity1_Dataset_Preparation.ipynb        # Activity 1 Jupyter Notebook
│   ├── student_performance_dataset.csv                 # Raw dataset
│   └── student_performance_prepared.csv                # Cleaned, encoded & scaled prepared dataset
│
└── activity_no2/                                       # ACTIVITY NO. 2: Model Training & Evaluation
    ├── CIE2_Activity2_Model_Prediction.ipynb           # Activity 2 Jupyter Notebook
    └── student_performance_prepared.csv                # Prepared input dataset
```

---

## 📝 Activity Breakdown

### 🔹 Activity No. 1: Dataset Preparation & Preprocessing
- **Folder**: `activity_no1/`
- **Notebook**: [`CIE2_Activity1_Dataset_Preparation.ipynb`](activity_no1/CIE2_Activity1_Dataset_Preparation.ipynb)
- **Topics Covered**:
  1. **Dataset Ingestion**: Loading raw student academic records.
  2. **Data Inspection**: Schema analysis, column types, descriptive statistics.
  3. **Quality Audit**: Missing value detection, whitespace anomaly detection, duplicate detection.
  4. **Data Cleaning**: String trimming, deduplication, median/mode imputation.
  5. **Exploratory Data Analysis (EDA)**: Target score distributions, demographic frequencies, correlation analysis.
  6. **Feature Engineering**: Outlier capping using $1.5 \times \text{IQR}$, One-Hot categorical encoding, and `StandardScaler` normalization.
  7. **Export**: Saved clean output as `student_performance_prepared.csv`.

---

### 🔹 Activity No. 2: Machine Learning Model Training & Evaluation
- **Folder**: `activity_no2/`
- **Notebook**: [`CIE2_Activity2_Model_Prediction.ipynb`](activity_no2/CIE2_Activity2_Model_Prediction.ipynb)
- **Topics Covered**:
  1. **Input Ingestion**: Loading preprocessed dataset.
  2. **Feature-Target Segregation**: Defining feature matrix $X$ and continuous target $y$ (`Final_Exam_Score`).
  3. **Train-Test Split**: 80% Training ($N=800$) and 20% Testing ($N=200$) with `random_state=42`.
  4. **Model Training**: Fitting an ordinary least squares Linear Regression model.
  5. **Model Evaluation**: Computing MAE, MSE, RMSE, and $R^2$ Score ($R^2 \approx 0.901$).
  6. **Visualization**: Actual vs. Predicted scatter plot with $y = x$ reference line.
  7. **Inference**: Scoring unseen hypothetical student profiles.

---

## 📈 Evaluation Results Summary

| Metric | Result | Meaning |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **~3.12** | Average deviation between actual and predicted exam score is ~3 points. |
| **Root Mean Squared Error (RMSE)** | **~3.95** | Low residual error confirming high model precision. |
| **$R^2$ Score** | **~0.901 (90.1%)** | Over 90% of the variance in exam scores is explained by the features. |

---

## 💻 Installation & Local Execution

```bash
# 1. Clone repository
git clone https://github.com/shravanibhandare9/CIE-2-activity-machine-learning.git
cd CIE-2-activity-machine-learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open Activity 1 Notebook
jupyter notebook activity_no1/CIE2_Activity1_Dataset_Preparation.ipynb

# 4. Open Activity 2 Notebook
jupyter notebook activity_no2/CIE2_Activity2_Model_Prediction.ipynb
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
