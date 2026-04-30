# 📊 Churn Prediction ML Pipeline

## 📌 Overview

This project implements an **end-to-end machine learning pipeline** to predict customer churn using the Telco Customer Churn dataset. The pipeline covers all stages of the ML workflow, from data preprocessing to model training, evaluation, and saving artifacts for future use.

---

## 🎯 Objective

The goal of this project is to:

* Predict whether a customer will churn (Yes/No)
* Build a reusable and automated ML pipeline
* Demonstrate production-level project structure

---

## 🗂️ Project Structure

```
churn-ml-pipeline/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── p3b_pipeline.ipynb
│
├── src/
│   └── train.py
│
├── model.pkl
├── scaler.pkl
├── README.md
```

---

## ⚙️ Pipeline Workflow

The project follows a structured pipeline:

1. **Data Ingestion**

   * Load dataset using Pandas

2. **Data Cleaning**

   * Remove unnecessary columns (e.g., `customerID`)
   * Handle missing values

3. **Feature Engineering**

   * Convert categorical variables using one-hot encoding

4. **Data Splitting**

   * Train-test split with stratification

5. **Feature Scaling**

   * Standardize numerical features using `StandardScaler`

6. **Model Training**

   * Train a Random Forest classifier

7. **Model Evaluation**

   * Accuracy Score
   * Classification Report
   * Confusion Matrix

8. **Model Saving**

   * Save trained model (`model.pkl`)
   * Save scaler (`scaler.pkl`)

---

## 🤖 Model Used

* **Random Forest Classifier**

  * Robust to overfitting
  * Handles tabular data effectively
  * Provides feature importance

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## ▶️ How to Run

### 📌 Run Training Script

```bash
python src/train.py
```

### 📌 Run Notebook (Optional)

Open:

```
notebooks/p3b_pipeline.ipynb
```

---

## 📦 Outputs

After running the pipeline:

* `model.pkl` → Trained ML model
* `scaler.pkl` → Preprocessing scaler

---

## 💼 Key Highlights

* End-to-end ML pipeline implementation
* Modular code using `train.py`
* Separation of notebook (analysis) and script (production)
* Ready for deployment

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Use advanced models (XGBoost, LightGBM)
* Deploy using Flask/FastAPI
* Add real-time prediction API

---

## 🧠 Learning Outcomes

* Understanding of full ML lifecycle
* Hands-on experience with preprocessing & modeling
* Knowledge of production-ready project structure

---

## 👤 Author

**Hemanth Gorijala**
AI/ML Trainee

---

## 📌 Note

This project is part of Day 3 training and demonstrates practical implementation of machine learning pipelines in a real-world scenario.
