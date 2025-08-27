# -*- coding: utf-8 -*-
# Dashboard interattiva Stroke (Kaggle) + tab "Model" con Logistic Regression

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

st.set_page_config(page_title="🧠 Stroke Dashboard", layout="wide")
st.title("🧠 Stroke Dashboard – Kaggle")

CSV_PATH = "data/healthcare-dataset-stroke-data.csv"

@st.cache_data
def load_data(path: str):
    if not os.path.exists(path):
        st.error(f"❌ File non trovato: `{path}`. Metti il CSV in `data/` con il nome esatto.")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "stroke" in df.columns:
        df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce").astype("Int64")
    for col in ("age", "avg_glucose_level", "bmi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data(CSV_PATH)
if df is None:
    st.stop()

st.success("✅ Dataset caricato")
st.caption(f"Shape: {df.shape[0]:,} righe × {df.shape[1]} colonne")
st.dataframe(df.head(10), width="stretch")

# ─────────────────────────────────────────────────────────
# Definizione tab: 4 tab, usati poi con "with tabX:"
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Distribuzioni", "Confronti categoria", "Relazioni continue", "Model"
])

# --- Tab 1: Distribuzioni ---
with tab1:
    c1, c2, c3 = st.columns(3)
    if "age" in df.columns:
        with c1:
            fig_age = px.histogram(df.dropna(subset=["age"]), x="age", nbins=30, title="Distribuzione Età")
            st.plotly_chart(fig_age, width="stretch")
    if "avg_glucose_level" in df.columns:
        with c2:
            fig_glu = px.histogram(df.dropna(subset=["avg_glucose_level"]), x="avg_glucose_level", nbins=30, title="Distribuzione Glucosio")
            st.plotly_chart(fig_glu, width="stretch")
    if "bmi" in df.columns:
        with c3:
            fig_bmi = px.histogram(df.dropna(subset=["bmi"]), x="bmi", nbins=30, title="Distribuzione BMI")
            st.plotly_chart(fig_bmi, width="stretch")

# --- Tab 2: Confronti categoria ---
with tab2:
    def rate_bar(col, title):
        if "stroke" not in df.columns or col not in df.columns:
            return None
        g = df[[col, "stroke"]].dropna()
        if g.empty:
            return None
        g = g.groupby(col)["stroke"].mean().reset_index()
        g["stroke_rate_%"] = g["stroke"] * 100
        return px.bar(g, x=col, y="stroke_rate_%", title=title, text="stroke_rate_%")

    for fig in [
        rate_bar("gender", "Tasso stroke per Genere"),
        rate_bar("smoking_status", "Tasso stroke per Fumo"),
        rate_bar("work_type", "Tasso stroke per Tipo di lavoro"),
        rate_bar("Residence_type", "Tasso stroke per Residenza"),
        rate_bar("ever_married", "Tasso stroke per Stato Civile"),
    ]:
        if fig:
            st.plotly_chart(fig, width="stretch")

# --- Tab 3: Relazioni continue (fix narwhals ShapeError) ---
with tab3:
    # Età vs Glucosio
    if all(c in df.columns for c in ["age", "avg_glucose_level", "stroke"]):
        df_plot = df.dropna(subset=["age", "avg_glucose_level", "stroke"]).copy()
        if not df_plot.empty:
            df_plot["stroke_str"] = df_plot["stroke"].astype(int).astype(str)
            fig = px.scatter(
                df_plot, x="age", y="avg_glucose_level",
                color="stroke_str", opacity=0.6,
                title="Età vs Glucosio (color: stroke)"
            )
            fig.update_layout(legend_title_text="stroke")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Nessun dato disponibile per Età vs Glucosio dopo il filtraggio dei NA.")

    # BMI vs Glucosio
    if all(c in df.columns for c in ["bmi", "avg_glucose_level", "stroke"]):
        df_plot2 = df.dropna(subset=["bmi", "avg_glucose_level", "stroke"]).copy()
        if not df_plot2.empty:
            df_plot2["stroke_str"] = df_plot2["stroke"].astype(int).astype(str)
            fig2 = px.scatter(
                df_plot2, x="bmi", y="avg_glucose_level",
                color="stroke_str", opacity=0.6,
                title="BMI vs Glucosio (color: stroke)"
            )
            fig2.update_layout(legend_title_text="stroke")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Nessun dato disponibile per BMI vs Glucosio dopo il filtraggio dei NA.")

# --- Tab 4: Model (Logistic Regression) ---
with tab4:
    st.subheader("Logistic Regression (baseline)")
    if "stroke" not in df.columns:
        st.warning("Colonna 'stroke' mancante.")
        st.stop()

    f = df.dropna(subset=["stroke"]).copy()
    f["stroke"] = f["stroke"].astype(int)

    vc = f["stroke"].value_counts()
    if len(vc) < 2 or vc.min() < 5:
        st.warning("Servono almeno 5 esempi per classe per allenare/valutare.")
        st.stop()

    y = f["stroke"]
    X = f.drop(columns=["stroke"])
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    num_cols = X.select_dtypes(include=["int64", "float64", "Int64", "Float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("AUC (ROC)", f"{auc:.3f}")
    with c2:
        st.metric("Average Precision (PR)", f"{ap:.3f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig_roc.update_layout(title=f"ROC Curve (AUC={auc:.3f})", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc, width="stretch")

    # PR
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
    fig_pr.update_layout(title=f"Precision–Recall Curve (AP={ap:.3f})",
                         xaxis_title="Recall", yaxis_title="Precision",
                         xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig_pr, width="stretch")

