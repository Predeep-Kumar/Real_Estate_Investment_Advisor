# 🏡 Real Estate Investment Advisor (ML + Streamlit + MLflow)

This repository contains a complete Machine Learning and Data Analytics pipeline for:

  - House Price Prediction (Regression Models)
  - Investment Decision Classification (Yes/No)
  - Model Comparison & Evaluation
  - MLflow Experiment Tracking
  - Streamlit Web Application

The project covers all stages:  
Preprocessing → Feature Engineering → EDA → Model Training → Model Comparison → MLflow Logging → Deployment

-------------------------------------------------------------


# 📁 Project Structure
```
Real_Estate_Investment_Advisor/
│
├── src/
│ ├── mlflow_init_experiment.py
│ ├── train_classification.py
│ ├── train_regression.py
│
├── config/
│ ├── best_classification_registered.json
│ ├── best_regression_registered.json
│ ├── mlflow_logged_classification_summary.json
│ ├── mlflow_logged_regression_summary.json
│ └── mlflow_all_models_config.json
│
├── data/
│ ├── raw/
│ │ └── india_housing_prices.csv
│ │
│ ├── processed/
│ ├── india_housing_cleaned.csv
│ └── india_housing_cleaned_base.csv
│
├── models/
│ ├── clf_logistic_regression_tuned.joblib
│ ├── clf_random_forest_tuned.joblib
│ ├── clf_xgboost_tuned.joblib
│ ├── reg_linear_regression.joblib
│ ├── reg_ridge_regression_tuned.joblib
│ ├── reg_random_forest_regressor_tuned.joblib
│ └── reg_xgboost_regressor_tuned.joblib
│
├── notebooks/
│ ├── 01_data_processing_and_feature_engineering.ipynb
│ ├── 02_eda_and_business_insights.ipynb
│ ├── 03_modeling_and_evaluation.ipynb
│ └── 04_model_comparison_and_selection.ipynb
│
├── mlruns/
│
├── reports/
│ ├── figures/
│ └── metrics/
│
├── start_mlflow.bat
├── start_mlflow.sh
├── app.py
├── requirements.txt
└── README.md
```
-------------------------------------------------------------


# ⚠️ Important Note — Model & MLflow Files

Large model artifacts and MLflow run files are NOT included in this GitHub repository  
because they exceed GitHub's file-size limits or are generated dynamically.

### Missing items (generated during training):

- MLflow model artifacts
- MLflow experiment run folders
- Best model registry files
- Large `.joblib` models (optional if removed)

👉 These files are created automatically when you run the training scripts or MLflow experiments.


OR you can download the complete project including models:

🔗 https://drive.google.com/drive/folders/138icC7Ed5h1Vs75T6zlb4sMeAJjpW3XB?usp=drive_link 

-------------------------------------------------------------


# 🧠 Training Pipeline

### ▶️ Run notebooks in order:

1) **01_data_processing_and_feature_engineering.ipynb**  
2) **02_eda_and_business_insights.ipynb**  
3) **03_modeling_and_evaluation.ipynb**  
4) **04_model_comparison_and_selection.ipynb**

### ▶️ Run notebooks in order:

1) **01_data_processing_and_feature_engineering.ipynb**  
2) **02_eda_and_business_insights.ipynb**  
3) **03_modeling_and_evaluation.ipynb**  
4) **04_model_comparison_and_selection.ipynb**

### ▶️ MLflow Training (after completing notebooks)

After preprocessing and EDA are complete, run MLflow setup and training scripts:

1) **start_mlflow.bat** (Windows) or **start_mlflow.sh** (Mac/Linux)  
2) **mlflow_init_experiment.py**
  ```python src/mlflow_init_experiment.py ```
4) **train_classification.py**
   ``` python src/train_regression.py --mlflow_uri http://127.0.0.1:5000``` 
6) **train_regression.py**
   ``` python src/train_regression.py --mlflow_uri http://127.0.0.1:5000```

### ▶️ Before running:

- Check dataset paths  
- Validate the `config/` directory  
- Ensure MLflow server is running (optional but recommended)

### ▶️ Output directories:

- **mlruns/** — MLflow experiment logs  
- **config/** — Best model details and metadata  
- **models/** — Optional: saved `.joblib` models

-------------------------------------------------------------

📌 *Note:*  
You only need to re-run the training pipeline if you want to modify models, update datasets, or retrain.  
Otherwise, the Streamlit app automatically loads the best registered models.

-------------------------------------------------------------

# 💻 Running the Streamlit App

### Go to the project root:
```Real_Estate_Investment_Advisor/```

### 1️⃣ Create Virtual Environment (Windows)
```
python -m venv venv
venv\Scripts\activate
```

### 1️⃣ Create Virtual Environment (Mac/Linux)
```
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Requirements
```pip install -r requirements.txt```


### 3️⃣ Run the App
```streamlit run app.py```
-------------------------------------------------------------


# 🚀 Features

### ✔️ House Price Prediction (Regression)
Models used:

- Linear Regression  
- Ridge Regression  
- Random Forest Regressor  
- XGBoost Regressor  

Includes:

- Error metrics (MAE, RMSE, R²)  
- Bar charts & insights  
- Auto-best model selection  

### ✔️ Investment Decision (Classification)

Models used:

- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

Includes:

- YES/NO decision  
- Confidence score  
- Accuracy, Precision, Recall, F1, AUC  
- Metrics shown in percentage  

### ✔️ Streamlit Dashboard

- Clean modern UI  
- Automatic metric updates  
- Visual charts and color-coded indicators  
- Model selection (manual or automatic)  

-------------------------------------------------------------

# 🔍 How It Works

### 🔹 Regression Mode
- Loads best regression model  
- Predicts future house price  
- Displays model metrics and charts  

### 🔹 Classification Mode
- Predicts investment decision (0/1 → YES/NO)  
- Shows confidence score  
- Displays classification metrics  

-------------------------------------------------------------

# 📊 Analytics Included

- Model comparison visualizations  
- Business insights from housing data  
- Regression & classification metric dashboards  
- Error analysis  

-------------------------------------------------------------

# ⭐ Future Enhancements

- Multi-city forecasting  
- SHAP interpretability dashboard  
- Real-time API-based pricing  
- Full Docker deployment  
- Automated retraining pipeline  
- Enhanced feature engineering  

-------------------------------------------------------------

# 🤝 Author

### Predeep Kumar  
Real Estate Investment Advisor — Machine Learning + Streamlit + MLflow Project

-------------------------------------------------------------
