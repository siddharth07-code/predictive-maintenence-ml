# ⚙️ Predictive Maintenance ML

A machine-learning based predictive maintenance system that analyzes machine operating parameters and predicts the probability of machine failure.

The project trains and compares **four machine-learning algorithms** and provides an interactive **Streamlit web application** where users can enter machine parameters and receive a failure prediction.

---

## 🚀 Features

- 📊 Exploratory Data Analysis of machine sensor data
- 🧹 Data preprocessing and feature engineering
- 🤖 Four machine-learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- 📈 Model performance comparison
- 🎯 Machine failure prediction
- 📊 Failure probability estimation
- 🌐 Interactive Streamlit frontend
- 🔍 Random Forest feature-importance analysis
- 💾 Saved trained models for direct application use

---

## 🧠 Project Workflow

```text
                    AI4I 2020 Dataset
                           │
                           ▼
                  Data Exploration
                           │
                           ▼
                   Data Preprocessing
                           │
                           ▼
                    Feature Selection
                           │
                           ▼
                    Train/Test Split
                           │
                           ▼
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          Logistic      Decision     Random
         Regression       Tree       Forest
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                        XGBoost
                           │
                           ▼
                   Model Comparison
                           │
                           ▼
                  Streamlit Web App
                           │
                           ▼
                  Machine Prediction
```

---

## 📊 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains **10,000 machine records** and includes machine operating parameters such as:

| Feature | Description |
|---|---|
| Air temperature | Air temperature in Kelvin |
| Process temperature | Process temperature in Kelvin |
| Rotational speed | Machine rotational speed in RPM |
| Torque | Machine torque in Nm |
| Tool wear | Tool wear time in minutes |
| Type | Machine/product type |
| Machine failure | Target variable |

The target variable is:

```text
0 → No machine failure
1 → Machine failure
```

The failure-specific columns were excluded from the initial predictive model to reduce the possibility of target leakage.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Decision Tree

A tree-based classifier that learns decision rules from the machine parameters.

### 3. Random Forest

An ensemble of multiple decision trees designed to improve robustness and predictive performance.

### 4. XGBoost

A gradient-boosting algorithm designed for high-performance classification on structured/tabular datasets.

---

## 📈 Model Performance

The initial evaluation produced the following results:

| Model | Accuracy | Failure Recall | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 82.50% | 82.35% | 0.242 |
| Decision Tree | 93.65% | **85.29%** | 0.477 |
| Random Forest | **97.90%** | 63.24% | 0.672 |
| XGBoost | **~99%** | 69.00% | **0.790** |

### Important observation

Accuracy alone is not sufficient for predictive maintenance because machine failures are relatively rare compared with normal operation.

Therefore, **precision, recall and F1-score for the failure class are also considered**.

The Decision Tree achieved the highest failure recall in the initial comparison, while XGBoost achieved the strongest overall F1-score and accuracy.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit frontend.

Users can enter:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

and select one of the four trained models.

The application provides:

- Machine status
- Failure probability
- Selected ML model
- Input parameter summary

### Application flow

```text
User Input
    ↓
Select ML Model
    ↓
Preprocessing
    ↓
Trained Model
    ↓
Prediction
    ↓
Failure Probability
    ↓
Machine Status
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## 📁 Project Structure

```text
predictive-maintenence-ml/
│
├── data/
│   └── ai4i2020.csv
│
├── pm.py
├── app.py
│
├── logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── xgboost.pkl
├── scaler.pkl
├── feature_names.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/siddharth07-code/predictive-maintenence-ml.git
```

```bash
cd predictive-maintenence-ml
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### macOS XGBoost requirement

If XGBoost reports a missing `libomp.dylib` error on macOS:

```bash
brew install libomp
```

---

## ▶️ Running the Project

### Train the models

Run:

```bash
python pm.py
```

This performs data preprocessing, trains all four models, evaluates their performance, generates feature-importance analysis, and saves the trained models.

### Launch the Streamlit application

Run:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔬 Future Improvements

Possible future extensions include:

- Hyperparameter optimization
- Cross-validation
- Improved handling of class imbalance
- Probability-threshold optimization
- SHAP-based model explainability
- Real-time sensor data integration
- IoT-based machine monitoring
- Cloud deployment
- Real-time machine failure alerts
- Historical maintenance dashboard

---

## 🎯 Project Objective

The primary objective of this project is to demonstrate how machine-learning techniques can be applied to machine operating data to identify potential failures before they occur.

The project also demonstrates the complete workflow from **dataset preprocessing and model training to model evaluation and deployment through a Python-based web interface**.

---

## 👨‍💻 Author

**Siddharth**

---

## 📄 Dataset Reference

AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository.