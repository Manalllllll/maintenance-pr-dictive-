import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Pump Health Monitoring Dashboard",
    page_icon="⚙️",
    layout="wide",
)

sns.set_theme(style="whitegrid")

DEFAULT_DATA_PATHS = [
    Path("/mnt/data/Centrifugal_pumps_measurements.xlsx"),
    Path("Centrifugal_pumps_measurements.xlsx"),
]
SENSOR_PRIORITY = ["velocity", "value_DEMODULATION", "value_ACC", "value_P2P", "valueTEMP"]
TIME_COLUMNS = ["year", "month", "day", "hour", "minute", "second"]
APP_ACCENT_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
}
h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
[data-testid=\"stMetricValue\"] {
    font-weight: 800;
}
.section-intro {
    font-size: 1rem;
    opacity: 0.9;
    margin-bottom: 0.4rem;
}
.hero-card {
    padding: 1rem 1.15rem;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px;
    margin-bottom: 0.75rem;
    background: rgba(250,250,250,0.55);
}
</style>
"""

st.markdown(APP_ACCENT_CSS, unsafe_allow_html=True)


def get_default_source_bytes():
    existing = next((p for p in DEFAULT_DATA_PATHS if p.exists()), None)
    if existing is None:
        return None, None
    return existing.read_bytes(), str(existing)


@st.cache_data
def load_data_from_bytes(file_bytes, source_name):
    if file_bytes is None:
        raise FileNotFoundError(
            "No dataset was found automatically. Upload `Centrifugal_pumps_measurements.xlsx` on the Upload & Predict page."
        )

    original_df = pd.read_excel(io.BytesIO(file_bytes))
    df = original_df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = len(original_df) - len(df)
    return df, source_name, removed_duplicates


@st.cache_data
def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if set(TIME_COLUMNS).issubset(out.columns):
        out["timestamp"] = pd.to_datetime(out[TIME_COLUMNS], errors="coerce")
    return out


@st.cache_resource
def train_models(df: pd.DataFrame):
    if "Machine_ID" not in df.columns:
        raise ValueError("The dataset must include a `Machine_ID` column for model training.")

    feature_df = df.drop(columns=["Machine_ID", "timestamp"], errors="ignore")
    numeric_features = feature_df.select_dtypes(include="number").copy()
    y = df["Machine_ID"].map({1: 0, 2: 1})

    valid_mask = y.notna()
    X = numeric_features.loc[valid_mask]
    y = y.loc[valid_mask]

    if X.empty:
        raise ValueError("No numeric feature columns are available for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    results = {
        "feature_columns": list(X.columns),
        "scaler": scaler,
        "rf_model": rf_model,
        "rf_accuracy": accuracy_score(y_test, rf_pred),
        "rf_report": classification_report(y_test, rf_pred, output_dict=True),
        "rf_confusion": confusion_matrix(y_test, rf_pred),
        "rf_importance": pd.DataFrame(
            {"feature": X.columns, "importance": rf_model.feature_importances_}
        ).sort_values("importance", ascending=False),
        "feature_summary": X.describe().transpose(),
        "feature_medians": X.median(),
    }

    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        results.update(
            {
                "xgb_model": xgb_model,
                "xgb_accuracy": accuracy_score(y_test, xgb_pred),
                "xgb_report": classification_report(y_test, xgb_pred, output_dict=True),
                "xgb_confusion": confusion_matrix(y_test, xgb_pred),
            }
        )
    except Exception as exc:
        results["xgb_error"] = str(exc)

    return results


@st.cache_data
def build_slider_config(feature_summary: pd.DataFrame):
    config = {}
    for feature, row in feature_summary.iterrows():
        min_val = float(row["min"])
        max_val = float(row["max"])
        default_val = float(row["50%"])
        if min_val == max_val:
            max_val = min_val + 1.0
        step = (max_val - min_val) / 100.0
        if step <= 0:
            step = 0.01
        config[feature] = {
            "min": min_val,
            "max": max_val,
            "value": default_val,
            "step": step,
        }
    return config


def get_loaded_dataset():
    uploaded_file = st.session_state.get("uploaded_file")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_name = uploaded_file.name
    else:
        file_bytes, source_name = get_default_source_bytes()

    df, source_name, removed_duplicates = load_data_from_bytes(file_bytes, source_name)
    return add_timestamp(df), source_name, removed_duplicates


def render_dashboard_page(df: pd.DataFrame, source_name: str, removed_duplicates: int):
    st.title("**Pump Health Monitoring Dashboard**")
    st.markdown(
        "<div class='hero-card'><b>Purpose:</b> Review data quality, sensor behavior, and model performance for the centrifugal pump dataset in one clean operational view.</div>",
        unsafe_allow_html=True,
    )
    st.success(f"Active dataset: {source_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Available Fields", len(df.columns))
    col3.metric("Duplicate Rows Removed", f"{removed_duplicates:,}")
    col4.metric("Detected Machines", df["Machine_ID"].nunique() if "Machine_ID" in df.columns else "N/A")

    with st.expander("**Dataset Preview**", expanded=True):
        st.caption("A quick look at the cleaned dataset currently loaded into the app.")
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("**Data Quality Overview**")
    st.markdown("<div class='section-intro'>Use this section to verify completeness and class balance before reviewing the charts.</div>", unsafe_allow_html=True)
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.write("**Missing Values by Column**")
        st.dataframe(df.isna().sum().rename("missing_values"), use_container_width=True)
    with summary_right:
        if "Machine_ID" in df.columns:
            machine_counts = df["Machine_ID"].value_counts().sort_index()
            st.write("**Record Count by Machine**")
            st.dataframe(machine_counts.rename("count"), use_container_width=True)

    st.subheader("**Exploratory Analysis**")
    st.markdown("<div class='section-intro'>These charts preserve the original notebook visuals while presenting them with cleaner labels.</div>", unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include="number")

    chart1, chart2 = st.columns(2)
    with chart1:
        fig, ax = plt.subplots(figsize=(12, 6))
        numeric_df.boxplot(rot=90, ax=ax)
        ax.set_title("Sensor Distribution Overview")
        ax.set_ylabel("Measured Value")
        st.pyplot(fig, clear_figure=True)

    with chart2:
        fig, ax = plt.subplots(figsize=(12, 8))
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig, clear_figure=True)

    chart3, chart4 = st.columns(2)
    with chart3:
        if "Machine_ID" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            machine_counts = df["Machine_ID"].value_counts().sort_index()
            ax.bar(machine_counts.index.astype(str), machine_counts.values)
            ax.set_title("Records per Machine")
            ax.set_xlabel("Machine ID")
            ax.set_ylabel("Number of Records")
            st.pyplot(fig, clear_figure=True)

    with chart4:
        if {"Machine_ID", "valueTEMP"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(12, 6))
            for machine_id in sorted(df["Machine_ID"].dropna().unique()):
                subset = df[df["Machine_ID"] == machine_id]
                ax.hist(subset["valueTEMP"], bins=20, alpha=0.5, label=f"Machine {machine_id}")
            ax.set_title("Temperature Distribution by Machine")
            ax.set_xlabel("Temperature Reading")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig, clear_figure=True)

    if "timestamp" in df.columns:
        st.subheader("**Sensor Trend Over Time**")
        st.markdown("<div class='section-intro'>Compare how a selected sensor evolves across the recorded timeline.</div>", unsafe_allow_html=True)
        time_df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        sensor_options = [c for c in SENSOR_PRIORITY if c in time_df.columns]
        if sensor_options:
            selected_sensor = st.selectbox("**Select a Sensor Signal**", sensor_options)
            fig, ax = plt.subplots(figsize=(12, 5))
            if "Machine_ID" in time_df.columns:
                for machine_id in sorted(time_df["Machine_ID"].dropna().unique()):
                    subset = time_df[time_df["Machine_ID"] == machine_id]
                    ax.plot(subset["timestamp"], subset[selected_sensor], label=f"Machine {machine_id}")
            else:
                ax.plot(time_df["timestamp"], time_df[selected_sensor])
            ax.set_title(f"{selected_sensor} Trend")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel(selected_sensor)
            ax.tick_params(axis="x", rotation=45)
            if "Machine_ID" in time_df.columns:
                ax.legend()
            st.pyplot(fig, clear_figure=True)

    st.subheader("**Model Performance Review**")
    st.info(
        "The original notebook used an invalid feature-label split. This app rebuilds the prediction workflow correctly with Machine 1 mapped to class 0 and Machine 2 mapped to class 1."
    )

    try:
        results = train_models(df)

        score_cols = st.columns(2)
        score_cols[0].metric("Random Forest Accuracy", f"{results['rf_accuracy']:.3f}")
        if "xgb_accuracy" in results:
            score_cols[1].metric("XGBoost Accuracy", f"{results['xgb_accuracy']:.3f}")
        else:
            score_cols[1].warning("XGBoost is not available in this runtime.")

        model_left, model_right = st.columns(2)
        with model_left:
            st.write("**Random Forest Classification Report**")
            st.dataframe(pd.DataFrame(results["rf_report"]).transpose(), use_container_width=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(results["rf_confusion"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Random Forest Confusion Matrix")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            st.pyplot(fig, clear_figure=True)

        with model_right:
            fig, ax = plt.subplots(figsize=(8, 5))
            top_n = results["rf_importance"].head(10).sort_values("importance")
            ax.barh(top_n["feature"], top_n["importance"])
            ax.set_title("Top 10 Feature Importances")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig, clear_figure=True)

            if "xgb_accuracy" in results:
                st.write("**XGBoost Classification Report**")
                st.dataframe(pd.DataFrame(results["xgb_report"]).transpose(), use_container_width=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(results["xgb_confusion"], annot=True, fmt="d", cmap="Greens", ax=ax)
                ax.set_title("XGBoost Confusion Matrix")
                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Actual Class")
                st.pyplot(fig, clear_figure=True)
            elif "xgb_error" in results:
                st.info(f"XGBoost section skipped: {results['xgb_error']}")

    except Exception as exc:
        st.error(f"The modeling section could not be completed: {exc}")


def render_upload_predict_page(df: pd.DataFrame, source_name: str):
    st.title("**Upload & Predict**")
    st.markdown(
        "<div class='hero-card'><b>Purpose:</b> Load a dataset and simulate a machine classification using adjustable sensor input values.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("**Dataset Input**")
    st.markdown("<div class='section-intro'>Upload an Excel file to replace the default dataset used across the app.</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("**Upload Pump Measurement Workbook (.xlsx)**", type=["xlsx"], key="uploaded_file")
    if uploaded_file is not None:
        st.success(f"Uploaded file ready: {uploaded_file.name}")
    else:
        st.info(f"Using current dataset: {source_name}")

    try:
        results = train_models(df)
        slider_config = build_slider_config(results["feature_summary"])
    except Exception as exc:
        st.error(f"Prediction controls are unavailable: {exc}")
        return

    st.subheader("**Manual Prediction Controls**")
    st.markdown("<div class='section-intro'>Adjust the sliders below to estimate whether the input pattern looks closer to Machine 1 or Machine 2.</div>", unsafe_allow_html=True)

    input_columns = st.columns(2)
    user_inputs = {}
    for idx, feature in enumerate(results["feature_columns"]):
        cfg = slider_config[feature]
        with input_columns[idx % 2]:
            user_inputs[feature] = st.slider(
                f"**{feature}**",
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["value"]),
                step=float(cfg["step"]),
            )

    input_df = pd.DataFrame([user_inputs])[results["feature_columns"]]
    scaled_input = results["scaler"].transform(input_df)
    pred = int(results["rf_model"].predict(scaled_input)[0])
    probs = results["rf_model"].predict_proba(scaled_input)[0]

    label_map = {0: "Machine 1 Pattern", 1: "Machine 2 Pattern"}

    st.subheader("**Prediction Summary**")
    result_left, result_right, result_third = st.columns(3)
    result_left.metric("Predicted Class", label_map.get(pred, str(pred)))
    result_right.metric("Probability: Machine 1", f"{probs[0]:.2%}")
    result_third.metric("Probability: Machine 2", f"{probs[1]:.2%}")

    with st.expander("**Selected Input Values**", expanded=False):
        st.dataframe(input_df.transpose().rename(columns={0: "selected_value"}), use_container_width=True)


page = st.sidebar.radio(
    "**Navigate**",
    ["Dashboard", "Upload & Predict"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Pump Analytics App**")
st.sidebar.caption("Review charts on the dashboard and use the second page for data upload and prediction controls.")

try:
    df, source_name, removed_duplicates = get_loaded_dataset()
except Exception as exc:
    if page == "Upload & Predict":
        st.title("**Upload & Predict**")
        st.markdown(
            "<div class='hero-card'><b>Purpose:</b> Start by uploading the centrifugal pump workbook, then use the prediction controls once the data is loaded.</div>",
            unsafe_allow_html=True,
        )
        st.file_uploader("**Upload Pump Measurement Workbook (.xlsx)**", type=["xlsx"], key="uploaded_file")
        st.warning(str(exc))
        st.stop()
    st.error(str(exc))
    st.stop()

if page == "Dashboard":
    render_dashboard_page(df, source_name, removed_duplicates)
else:
    render_upload_predict_page(df, source_name)

st.markdown("---")
st.markdown(
    "Run locally with `streamlit run streamlit_dashboard.py`. The app will automatically use `Centrifugal_pumps_measurements.xlsx` when it is available, or you can upload a workbook on the **Upload & Predict** page."
)
