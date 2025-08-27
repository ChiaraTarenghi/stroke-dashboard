# 🧠 Stroke Prediction Dashboard

🔗 **Live demo**: [Stroke Dashboard on Streamlit Cloud](https://stroke-dashboard-nglzhpcr74jh6ipja8x2sh.streamlit.app/)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Interactive dashboard built with **Streamlit**, **Plotly**, and **scikit-learn** to explore and model stroke prediction data.

---

## 📊 Dataset
Dataset: **Stroke Prediction Dataset** from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 5,110 patient records
- Features: demographic, lifestyle, and health risk factors
- Target: `stroke` (1 = stroke, 0 = no stroke)

⚠️ **Note**: Due to file size/licensing, the dataset is **not included** in this repository.  
To run the dashboard locally, download the CSV from Kaggle and place it in:

stroke-dashboard/data/healthcare-dataset-stroke-data.csv
---

## 🚀 Features

### Data Exploration
- KPI summary (population, average age, BMI, stroke rate)
- Interactive filters by gender, smoking, work type, residence, marital status, age, glucose, BMI
- Distribution plots (age, BMI, glucose)
- Stroke rate comparison across categorical variables
- Scatter plots of continuous variables

### Modeling
- Logistic Regression (scikit-learn) trained on filtered data
- Class balancing with `class_weight="balanced"`
- Evaluation metrics:
  - ROC Curve + AUC
  - Precision–Recall Curve + Average Precision
- Feature importance table with download option

---

## 🖥️ Demo

Run locally with Streamlit:

```bash
# clone repo
git clone https://github.com/<your-username>/stroke-dashboard.git
cd stroke-dashboard

# create virtual env
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run app
streamlit run app/dashboard.py
