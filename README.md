# 🩺 Diabetes Risk Predictor

A machine learning web app that predicts diabetes risk based on health metrics.

🔗 **Live Demo:** https://spoorthy-diabetes-prediction.streamlit.app/

---

## 📌 About the Project

This project uses the **Pima Indians Diabetes Dataset** (UCI Machine Learning Repository) to predict whether a patient is at risk of diabetes based on diagnostic health measurements.

Built as part of my data science portfolio to demonstrate end-to-end ML project skills — from raw data to deployed web app.

---

## 🔍 Key Findings

- **Glucose** was the strongest predictor of diabetes (correlation: 0.49)
- Dataset had hidden missing values disguised as zeros in 5 columns — handled via median imputation
- Dataset is imbalanced (65% non-diabetic, 35% diabetic) — used ROC-AUC as primary metric
- Despite trying complex models, **Logistic Regression performed best** (76% accuracy, 0.73 ROC-AUC) — showing that model complexity should match data size
- SHAP analysis confirmed high glucose values consistently pushed predictions toward diabetes risk

---

## 📊 Model Comparison

| Model | Accuracy | ROC-AUC | Recall (Diabetic) |
|---|---|---|---|
| Logistic Regression | 76% | 0.732 | 0.64 |
| Random Forest | 75% | 0.735 | 0.67 |
| XGBoost | 72% | 0.706 | 0.65 |

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualizations
- **Scikit-learn** — ML models
- **XGBoost** — gradient boosting
- **SHAP** — model explainability
- **Streamlit** — web app deployment

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structurediabetes-prediction/
│
├── app.py               # Streamlit web app
├── diabetes.csv         # Dataset
├── requirements.txt     # Dependencies
└── README.md            # Project documentation---
## 👩‍💻 Author
**Spoorthy** — B.Tech CSE (Data Science Minor)
[GitHub]([https://github.com/](https://github.com/yourusername)SpoorthySM) | [LinkedIn]([https://linkedin.com/in/](https://linkedin.com/in/yourusername)spoorthysree)# diabetes-prediction
