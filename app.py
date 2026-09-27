from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Threshold F1-optimal dari Bagian 17 notebook, dipakai konsisten untuk keputusan kelas.
FALLBACK_THRESHOLD = 0.35

# Batas segmen risiko mengikuti Bagian 21.
SEGMENT_BOUNDS = (0.30, 0.60)

SERVICE_FLAG_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

SERVICE_OPTIONS = {
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
}

BINARY_OPTIONS = ["No", "Yes"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

BLUE, ORANGE, GREEN, RED, TEAL, PURPLE = "#4682B4", "#D2691E", "#2E8B57", "#CD5C5C", "#008080", "#6A5ACD"


def tenure_group(t):
    if t <= 12:
        return "0-12"
    if t <= 24:
        return "13-24"
    if t <= 48:
        return "25-48"
    return "49+"


def risk_segment(p):
    low, high = SEGMENT_BOUNDS
    if p < low:
        return "Low"
    if p < high:
        return "Medium"
    return "High"


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    # 11 pelanggan dengan tenure 0 punya TotalCharges kosong; tagihan kumulatifnya memang 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["total_services"] = (df[SERVICE_FLAG_COLS] == "Yes").sum(axis=1)
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    df["tenure_group"] = df["tenure"].apply(tenure_group)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(ARTIFACT_DIR / "churn_model.pkl")


@st.cache_data(show_spinner=False)
def load_tables():
    return joblib.load(ARTIFACT_DIR / "tables.pkl")


@st.cache_data(show_spinner=False)
def get_test_predictions():
    # Split yang sama seperti notebook supaya kurva dan segmen tetap konsisten.
    model = load_model()
    df = load_data()
    feature_cols = list(model.feature_names_in_)
    X = df[feature_cols]
    y = df["Churn_binary"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_proba = model.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_proba


def churn_rate_by(df, col):
    return df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)


def bar_churn_rate(df, col, color=BLUE, order=None):
    rate = churn_rate_by(df, col)
    if order is not None:
        rate = rate.reindex(order)
    rate = rate.sort_values(ascending=False)
    fig = px.bar(x=rate.index.astype(str), y=rate.values, color=rate.index.astype(str),
                 color_discrete_sequence=px.colors.qualitative.Bold, labels={"x": col, "y": "Churn rate (%)"})
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, title=f"Churn rate by {col}", margin=dict(t=50, b=10))
    return fig


def render_overview(df, tables, threshold):
    churn_counts = df["Churn"].value_counts()
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    roc_auc = tables["comparison_df"]["Test ROC-AUC"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pelanggan", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate:.2f}%")
    c3.metric("ROC-AUC (test)", f"{roc_auc:.4f}")
    c4.metric("Threshold keputusan", f"{threshold:.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(names=churn_counts.index, values=churn_counts.values, hole=0.5,
                     color=churn_counts.index, color_discrete_map={"No": BLUE, "Yes": RED},
                     title="Distribusi churn")
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        fig = px.bar(x=churn_counts.index, y=churn_counts.values, color=churn_counts.index,
                     color_discrete_map={"No": BLUE, "Yes": RED},
                     labels={"x": "Churn", "y": "Jumlah pelanggan"}, title="Jumlah pelanggan per kelas")
        fig.update_traces(texttemplate="%{y}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')


def render_analysis(df):
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                "OnlineSecurity", "SeniorCitizen", "Partner", "Dependents", "gender"]
    picked = st.selectbox("Pilih variabel", cat_cols)
    order = ["0-12", "13-24", "25-48", "49+"] if picked == "tenure_group" else None
    st.plotly_chart(bar_churn_rate(df, picked, order=order), width='stretch')

    st.subheader("Interaksi Contract x InternetService")
    pivot = df.pivot_table(index="Contract", columns="InternetService", values="Churn",
                           aggfunc=lambda s: (s == "Yes").mean() * 100)
    fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                    labels=dict(color="Churn rate (%)"))
    fig.update_layout(title="Churn rate (%): Contract x InternetService")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Distribusi numerik berdasarkan churn")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.box(df, x="Churn", y="tenure", color="Churn", category_orders={"Churn": ["No", "Yes"]},
                     color_discrete_map={"No": BLUE, "Yes": RED}, labels={"tenure": "Tenure (bulan)"})
        fig.update_layout(showlegend=False, title="Tenure by Churn")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn", category_orders={"Churn": ["No", "Yes"]},
                     color_discrete_map={"No": BLUE, "Yes": RED}, labels={"MonthlyCharges": "Monthly Charges ($)"})
        fig.update_layout(showlegend=False, title="MonthlyCharges by Churn")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Tingkat churn per kelompok tenure")
    st.plotly_chart(bar_churn_rate(df, "tenure_group", order=["0-12", "13-24", "25-48", "49+"]),
                    width='stretch')


def render_statistics(df, tables):
    numeric_for_corr = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    corr = df[numeric_for_corr + ["Churn_binary"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Correlation matrix (numerik + Churn)")
    st.plotly_chart(fig, width='stretch')

    assoc = tables["assoc_df"].sort_values("cramers_v")
    fig = px.bar(assoc, x="cramers_v", y="feature", orientation="h",
                 color="significant_at_0.05",
                 color_discrete_map={True: BLUE, False: RED},
                 labels={"cramers_v": "Cramer's V", "feature": "", "significant_at_0.05": "Signifikan (p<0.05)"})
    fig.update_layout(title="Kekuatan asosiasi dengan churn (Cramer's V)")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(tables["assoc_df"], width='stretch', hide_index=True)


def render_model_performance(tables, threshold):
    st.subheader("Perbandingan model")
    st.dataframe(tables["comparison_df"], width='stretch', hide_index=True)

    X_test, y_test, y_proba = get_test_predictions()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc_score(y_test, y_proba):.3f}",
                                 line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        ap = average_precision_score(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, name=f"AP = {ap:.3f}", line=dict(color=GREEN, width=3)))
        fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                      annotation_text=f"No-skill ({y_test.mean():.3f})")
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Trade-off threshold")
    tdf = tables["threshold_df"]
    fig = go.Figure()
    for col, color in [("precision", BLUE), ("recall", ORANGE), ("f1", GREEN)]:
        fig.add_trace(go.Scatter(x=tdf["threshold"], y=tdf[col], name=col.capitalize(),
                                 mode="lines+markers", line=dict(color=color)))
    fig.add_vline(x=threshold, line_dash="dash", line_color=RED,
                  annotation_text=f"F1-optimal ({threshold:.2f})")
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="default 0.5")
    fig.update_layout(title="Precision / Recall / F1 vs threshold",
                      xaxis_title="Threshold", yaxis_title="Score")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(tdf, width='stretch', hide_index=True)


def render_interpretability(tables):
    coef = tables["coef_df"].sort_values("coefficient")
    fig = px.bar(coef, x="coefficient", y="feature", orientation="h",
                 color=coef["coefficient"] > 0,
                 color_discrete_map={True: RED, False: BLUE},
                 labels={"coefficient": "Koefisien (log-odds)", "feature": ""})
    fig.update_layout(title="Koefisien Logistic Regression (merah = menaikkan odds churn)")
    st.plotly_chart(fig, width='stretch')

    perm = tables["perm_df"].head(12).sort_values("importance_mean")
    fig = px.bar(perm, x="importance_mean", y="feature", orientation="h",
                 error_x="importance_std", color_discrete_sequence=[GREEN],
                 labels={"importance_mean": "Penurunan ROC-AUC", "feature": ""})
    fig.update_layout(title="Permutation importance")
    st.plotly_chart(fig, width='stretch')


def render_risk(tables):
    summary = tables["segment_summary"].reset_index()
    st.dataframe(summary, width='stretch', hide_index=True)

    X_test, y_test, y_proba = get_test_predictions()
    risk_df = X_test.copy()
    risk_df["churn_probability"] = y_proba
    risk_df["actual_churn"] = y_test.values
    risk_df["risk_segment"] = risk_df["churn_probability"].apply(risk_segment)

    order = ["Low", "Medium", "High"]
    calib = risk_df.groupby("risk_segment").agg(
        avg_predicted=("churn_probability", "mean"),
        actual=("actual_churn", "mean"),
    ).reindex(order).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=calib["risk_segment"], y=calib["avg_predicted"], name="Prediksi rata-rata",
                         marker_color=BLUE))
    fig.add_trace(go.Bar(x=calib["risk_segment"], y=calib["actual"], name="Churn aktual",
                         marker_color=RED))
    fig.update_layout(barmode="group", title="Kalibrasi: probabilitas prediksi vs churn aktual",
                      yaxis_title="Rate")
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Karakteristik segmen (rata-rata)")
        st.dataframe(risk_df.groupby("risk_segment")[["tenure", "MonthlyCharges", "total_services"]]
                     .mean().round(2).reindex(order), width='stretch')
    with col_b:
        st.caption("Bauran kontrak per segmen (%)")
        mix = pd.crosstab(risk_df["risk_segment"], risk_df["Contract"], normalize="index") * 100
        st.dataframe(mix.round(1).reindex(order), width='stretch')


def render_prediction(model, threshold):
    feature_cols = list(model.feature_names_in_)
    with st.form("predict_form"):
        st.markdown("**Demografi**")
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Female", "Male"])
        senior = d2.selectbox("SeniorCitizen", [0, 1])
        partner = d3.selectbox("Partner", BINARY_OPTIONS)
        dependents = d4.selectbox("Dependents", BINARY_OPTIONS)

        st.markdown("**Layanan**")
        s1, s2, s3 = st.columns(3)
        phone = s1.selectbox("PhoneService", SERVICE_OPTIONS["PhoneService"])
        multiple = s2.selectbox("MultipleLines", SERVICE_OPTIONS["MultipleLines"])
        internet = s3.selectbox("InternetService", SERVICE_OPTIONS["InternetService"])
        s4, s5, s6 = st.columns(3)
        security = s4.selectbox("OnlineSecurity", SERVICE_OPTIONS["OnlineSecurity"])
        backup = s5.selectbox("OnlineBackup", SERVICE_OPTIONS["OnlineBackup"])
        protection = s6.selectbox("DeviceProtection", SERVICE_OPTIONS["DeviceProtection"])
        s7, s8, s9 = st.columns(3)
        support = s7.selectbox("TechSupport", SERVICE_OPTIONS["TechSupport"])
        tv = s8.selectbox("StreamingTV", SERVICE_OPTIONS["StreamingTV"])
        movies = s9.selectbox("StreamingMovies", SERVICE_OPTIONS["StreamingMovies"])

        st.markdown("**Kontrak & tagihan**")
        b1, b2, b3 = st.columns(3)
        contract = b1.selectbox("Contract", CONTRACT_OPTIONS)
        paperless = b2.selectbox("PaperlessBilling", BINARY_OPTIONS)
        payment = b3.selectbox("PaymentMethod", PAYMENT_OPTIONS)
        n1, n2, n3 = st.columns(3)
        tenure = n1.slider("Tenure (bulan)", 0, 72, 12)
        monthly = n2.number_input("MonthlyCharges ($)", 0.0, 200.0, 70.0, step=0.05)
        total = n3.number_input("TotalCharges ($)", 0.0, 10000.0, 840.0, step=1.0)

        submitted = st.form_submit_button("Prediksi", width='stretch')

    if not submitted:
        return

    services = {
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
    }
    row = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        "total_services": sum(1 for v in services.values() if v == "Yes"),
    }
    # Pipeline menyimpan urutan kolom saat training; baris input harus mengikuti urutan itu.
    X_input = pd.DataFrame([row])[feature_cols]

    proba = float(model.predict_proba(X_input)[:, 1][0])
    segment = risk_segment(proba)
    label = "CHURN" if proba >= threshold else "NO CHURN"

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilitas churn", f"{proba:.1%}")
    c2.metric(f"Kelas @ {threshold:.2f}", label)
    c3.metric("Segmen risiko", segment)

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100, number={"suffix": "%"},
        title={"text": "Probabilitas churn"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RED if proba >= threshold else BLUE},
            "steps": [
                {"range": [0, SEGMENT_BOUNDS[0] * 100], "color": "#D6EAF8"},
                {"range": [SEGMENT_BOUNDS[0] * 100, SEGMENT_BOUNDS[1] * 100], "color": "#FDEBD0"},
                {"range": [SEGMENT_BOUNDS[1] * 100, 100], "color": "#FADBD8"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=320, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width='stretch')


def main():
    st.set_page_config(page_title="Telco Customer Churn", layout="wide")

    if not (ARTIFACT_DIR / "churn_model.pkl").exists() or not (ARTIFACT_DIR / "tables.pkl").exists():
        st.error("Artefak model belum tersedia. Jalankan `prediction.ipynb` sampai selesai "
                 "agar folder `artifacts/` terbentuk, lalu muat ulang halaman ini.")
        st.stop()

    model = load_model()
    tables = load_tables()
    df = load_data()
    threshold = tables.get("final_threshold", FALLBACK_THRESHOLD)
    final_name = tables.get("final_name", "Model final")

    st.title("Telco Customer Churn Dashboard")
    st.caption(f"Model: {final_name} | Threshold: {threshold:.2f}")

    tabs = st.tabs(["Overview", "Analisis", "Statistik", "Performa Model",
                    "Interpretasi", "Risiko", "Prediksi"])

    with tabs[0]:
        render_overview(df, tables, threshold)
    with tabs[1]:
        render_analysis(df)
    with tabs[2]:
        render_statistics(df, tables)
    with tabs[3]:
        render_model_performance(tables, threshold)
    with tabs[4]:
        render_interpretability(tables)
    with tabs[5]:
        render_risk(tables)
    with tabs[6]:
        render_prediction(model, threshold)


if __name__ == "__main__":
    main()
