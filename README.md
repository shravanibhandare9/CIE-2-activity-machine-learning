# 🎓 Machine Learning CIE-2 Activity: Dataset Preparation, Exploratory Data Analysis & Data Preprocessing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Continuous Internal Evaluation (CIE-2) Academic Activity**  
> **Course**: Machine Learning Laboratory / Continuous Internal Evaluation  
> **Topic**: End-to-End Tabular Data Pipeline: Ingestion, Inspection, Exploratory Data Analysis (EDA), Preprocessing, and Baseline Regression Modeling.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Architecture](#-repository-architecture)
- [Dataset Description & Data Dictionary](#-dataset-description--data-dictionary)
- [Methodology & Pipeline Workflow](#-methodology--pipeline-workflow)
  - [1. Dataset Preparation & Inspection](#1-dataset-preparation--inspection)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Data Preprocessing & Cleaning](#3-data-preprocessing--cleaning)
  - [4. Baseline Model Evaluation](#4-baseline-model-evaluation)
- [Baseline Model Results](#-baseline-model-results)
- [Installation & How to Run](#-installation--how-to-run)
  - [Option A: Running Locally (Jupyter Notebook or Script)](#option-a-running-locally)
  - [Option B: Running on Google Colab](#option-b-running-on-google-colab)
- [Step-by-Step GitHub Upload Guide](#-step-by-step-github-upload-guide)
  - [Method 1: Using Git Command Line (Recommended)](#method-1-using-git-command-line-recommended)
  - [Method 2: Using GitHub Web Interface (Drag and Drop)](#method-2-using-github-web-interface)
- [License](#-license)

---

## 🚀 Project Overview

The objective of this college activity is to perform an end-to-end data science and machine learning workflow on a real-world multi-attribute dataset:
1. **Dataset Preparation**: Loading the raw dataset, inspecting structural properties, identifying data types, and auditing real-world imperfections (missing values, whitespace discrepancies, duplicate rows).
2. **Exploratory Data Analysis (EDA)**: Visualizing numerical distributions (univariate), categorical frequencies, bivariate relationships with student achievement, multivariate correlation heatmaps, and outlier detection with the $1.5 \times \text{IQR}$ rule.
3. **Data Preprocessing**: Implementing data cleaning (string trimming, deduplication), median/mode imputation, IQR-based Winsorization (capping), One-Hot categorical encoding, and feature standardization via `StandardScaler`.
4. **Model Baseline & Verification**: Splitting data into an 80:20 Train-Test split and training a Linear Regression baseline model to predict `Final_Exam_Score`, demonstrating high predictive accuracy ($R^2 \approx 0.90$).

---

## 📁 Repository Architecture

```text
CIE-2-activity-machine-learning/
├── README.md                                           # Comprehensive project documentation
├── requirements.txt                                    # Python dependencies list
├── .gitignore                                          # Git ignore file for Python/Jupyter
├── run_pipeline.py                                     # Standalone Python script reproducing entire workflow
├── data/
│   ├── student_performance_dataset.csv                 # Raw dataset (with intentional nulls, duplicates & outliers)
│   └── student_performance_prepared.csv                # Fully cleaned, encoded & scaled prepared dataset
└── notebooks/
    └── CIE2_Dataset_Preparation_EDA_Preprocessing.ipynb # Interactive, fully documented Jupyter Notebook
```

---

## 📊 Dataset Description & Data Dictionary

The project utilizes the **Student Academic Performance & Learning Analytics Dataset** containing **1,000 unique student records** and 12 distinct attributes:

| Feature Name | Data Type | Role | Description & Domain Values |
| :--- | :--- | :--- | :--- |
| `Student_ID` | String / Object | Identifier | Unique student tracking code (e.g., `STU1001`) |
| `Age` | Integer | Feature | Age of the student (15 to 24 years) |
| `Gender` | Categorical | Feature | Student gender (`Male`, `Female`, `Other`) |
| `Parental_Education` | Categorical | Feature | Highest parent education (`High School`, `Associate Degree`, `Bachelor's`, `Master's`, `Doctorate`) |
| `Study_Hours_Per_Week` | Float | Feature | Average weekly study hours (Continuous, contains ~3.5% missing values & upper outliers) |
| `Attendance_Rate` | Float | Feature | Class attendance percentage (40.0% to 100.0%) |
| `Tutoring_Sessions` | Integer / Float | Feature | Monthly extra tutoring sessions (0 to 6, contains ~2.5% missing values) |
| `Extracurricular_Activities` | Categorical | Feature | Participation in sports/clubs (`Yes`, `No`) |
| `Internet_Access` | Categorical | Feature | Home high-speed internet availability (`Yes`, `No`) |
| `Sleep_Hours_Per_Night` | Float | Feature | Average nightly sleep hours (4.0 to 10.0 hrs) |
| `Previous_Exam_Score` | Float | Feature | Past semester exam performance (35.0 to 100.0) |
| **`Final_Exam_Score`** | Float | **Target** | Comprehensive final exam score (Continuous: 25.0 to 100.0) |

---

## 🔬 Methodology & Pipeline Workflow

### 1. Dataset Preparation & Inspection
- **Ingestion**: Ingesting `student_performance_dataset.csv` into a Pandas DataFrame.
- **Structural Overview**: Verification of shape ($1006 \times 12$ rows including duplicates), columns, and memory footprint using `df.info()`.
- **Quality Audit**:
  - `Study_Hours_Per_Week`: 35 missing values (~3.5%).
  - `Tutoring_Sessions`: 25 missing values (~2.5%).
  - Detected 6 exact duplicate rows.
  - Detected leading/trailing whitespace in string entries (e.g. `" Yes "`, `"Female "`).

### 2. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Histograms with Kernel Density Estimation (KDE) demonstrating that `Final_Exam_Score` follows a near-normal bell curve centered around 72.0.
- **Categorical Distributions**: Count plots showing balanced representations across genders and parental education tiers.
- **Bivariate Analysis**: Scatter plots with linear regression trendlines confirming strong positive correlation between `Study_Hours_Per_Week` ($r = 0.72$), `Previous_Exam_Score` ($r = 0.68$), and `Final_Exam_Score`.
- **Multivariate Analysis**: Correlation heatmap assessing inter-feature collinearity.
- **Outlier Detection**: Boxplots and $1.5 \times \text{IQR}$ thresholding identifying extreme values ($>50$ hours/week).

### 3. Data Preprocessing & Cleaning
1. **String Normalization**: Applying `.str.strip()` across all text columns to eliminate whitespace inconsistencies.
2. **Deduplication**: Executing `df.drop_duplicates()` to remove 6 redundant rows.
3. **Missing Value Imputation**:
   - Skewed continuous feature (`Study_Hours_Per_Week`) imputed using **Median** ($18.0$ hrs).
   - Discrete feature (`Tutoring_Sessions`) imputed using **Mode** ($2$ sessions).
4. **Outlier Treatment (Winsorization)**: Capping `Study_Hours_Per_Week` within $[0, Q_3 + 1.5 \times \text{IQR}]$ to preserve records without distorting gradient descent.
5. **Categorical Encoding**: One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) to convert nominal categories while preventing the dummy variable trap.
6. **Feature Scaling**: Applying `StandardScaler` to continuous predictors ($z = \frac{x - \mu}{\sigma}$), ensuring zero mean and unit variance.
7. **Prepared Data Export**: Saving clean output to `student_performance_prepared.csv`.

### 4. Baseline Model Evaluation
- 80% Training set ($800$ rows) and 20% Test set ($200$ rows) via `train_test_split(random_state=42)`.
- Fitted an ordinary least squares `LinearRegression` model.

---

## 📈 Baseline Model Results

| Evaluation Metric | Baseline Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **~3.12** | Average deviation of predicted score from actual score is only ~3 points. |
| **Root Mean Squared Error (RMSE)** | **~3.95** | Penalizes larger errors; confirms low variance in residuals. |
| **$R^2$ Score (Coefficient of Determination)** | **~0.901 (90.1%)** | Over 90% of variance in final scores is explained by the preprocessed features. |

---

## 💻 Installation & How to Run

### Option A: Running Locally

#### 1. Clone or Download the Repository
```bash
git clone https://github.com/shravanibhandare9/CIE-2-activity-machine-learning.git
cd CIE-2-activity-machine-learning
```

#### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/CIE2_Dataset_Preparation_EDA_Preprocessing.ipynb
```
*Or open the folder in **VS Code** and select the Jupyter Notebook directly.*

#### 5. (Alternative) Run the Python Pipeline Script
```bash
python run_pipeline.py
```

---

### Option B: Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** > **Upload Notebook** and upload `notebooks/CIE2_Dataset_Preparation_EDA_Preprocessing.ipynb`.
3. In the left sidebar, click the 📁 **Files** icon and upload `data/student_performance_dataset.csv`.
4. Click **Runtime** > **Run all** (`Ctrl + F9`).

---

## 📤 Step-by-Step GitHub Upload Guide

Follow either of the methods below to upload this project to your GitHub account.

### Method 1: Using Git Command Line (Recommended)

1. Open **Terminal** (macOS/Linux) or **PowerShell / Git Bash** (Windows).
2. Navigate to your project folder:
   ```bash
   cd path/to/CIE-2-activity-machine-learning
   ```
3. Initialize Git:
   ```bash
   git init
   ```
4. Add all files to staging:
   ```bash
   git add .
   ```
5. Commit the files:
   ```bash
   git commit -m "feat: complete CIE-2 dataset preparation, EDA, preprocessing and notebook"
   ```
6. Create a new repository on [GitHub](https://github.com/new):
   - Name: `CIE-2-activity-machine-learning`
   - Description: *Machine Learning CIE 2 activity: Dataset preparation, exploratory data analysis and data preprocessing using Python.*
   - Visibility: **Public**
   - **Do NOT** initialize with README or .gitignore (we already created them).
7. Link your local repo to GitHub and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/shravanibhandare9/CIE-2-activity-machine-learning.git
   git push -u origin main
   ```

---

### Method 2: Using GitHub Web Interface

If you do not have Git installed on your computer:
1. Go to [GitHub](https://github.com/) and click the **+** icon in the top right > **New repository**.
2. Name it `CIE-2-activity-machine-learning`, set it to **Public**, and click **Create repository**.
3. On the repository page, click the link: **"uploading an existing file"**.
4. Drag and drop the following files and folders:
   - `README.md`
   - `requirements.txt`
   - `.gitignore`
   - `run_pipeline.py`
   - `data/` folder (containing both CSVs)
   - `notebooks/` folder (containing the `.ipynb` file)
5. Under **Commit changes**, type: `Initial commit for CIE-2 ML activity`.
6. Click **Commit changes**. Your repository is now live!

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
