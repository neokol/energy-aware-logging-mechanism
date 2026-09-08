import streamlit as st
import requests
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Experiment Results", layout="wide")
st.title("📊 Experiment Results")
st.markdown("Browse and compare all completed experiments.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Current Backend:")
    st.code(API_URL, language="text")
    if st.button("🔄 Refresh"):
        st.rerun()

# --- FETCH EXPERIMENTS + DATASETS ---
try:
    resp = requests.get(f"{API_URL}/experiments/")
    experiments = resp.json() if resp.status_code == 200 else []
except Exception:
    experiments = []

try:
    ds_resp = requests.get(f"{API_URL}/datasets")
    datasets = ds_resp.json().get("datasets", []) if ds_resp.status_code == 200 else []
except Exception:
    datasets = []

try:
    m_resp = requests.get(f"{API_URL}/models")
    models = m_resp.json() if m_resp.status_code == 200 else []
except Exception:
    models = []

if not experiments:
    st.warning("No experiments found. Run some experiments first.")
    st.stop()

df = pd.DataFrame(experiments)

# Join ai_model from datasets
if datasets:
    ds_df = pd.DataFrame(datasets)[["id", "ai_model", "filename"]].rename(
        columns={"id": "dataset_id"}
    )
    df = df.merge(ds_df, on="dataset_id", how="left")
else:
    df["ai_model"] = "Unknown"
    df["filename"] = ""

# Per-experiment architecture: prefer the uploaded model's type (a batch mixes
# architectures under one dataset), fall back to the dataset's ai_model.
if models and "model_id" in df.columns:
    m_df = pd.DataFrame(models)[["id", "model_type"]].rename(
        columns={"id": "model_id", "model_type": "arch"}
    )
    df = df.merge(m_df, on="model_id", how="left")
    df["arch"] = df["arch"].fillna(df["ai_model"])
else:
    df["arch"] = df["ai_model"]

# Normalize column types
for col in ["energy_consumed_kwh", "emissions_kg", "latency_seconds", "accuracy", "cpu_energy_kwh", "ram_energy_kwh", "throughput_samples_per_sec"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["created_at_str"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")

# --- SUMMARY METRICS ---
st.subheader("Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Experiments", len(df))
col2.metric("FP32 Runs", int((df["precision"] == "FP32").sum()))
col3.metric("INT8 Runs", int((df["precision"] == "INT8").sum()))
col4.metric("MLP Runs", int((df["arch"] == "MLP").sum()))
col5.metric("CNN Runs", int((df["arch"] == "CNN").sum()))

st.divider()

# --- FILTERS ---
st.subheader("Filter & Explore")
fcol1, fcol2, fcol3 = st.columns(3)
precision_filter = fcol1.multiselect("Precision", options=["FP32", "INT8"], default=["FP32", "INT8"])
model_filter = fcol2.multiselect("Architecture", options=df["arch"].dropna().unique().tolist(), default=df["arch"].dropna().unique().tolist())
dataset_filter = fcol3.multiselect("Dataset", options=df["filename"].dropna().unique().tolist(), default=df["filename"].dropna().unique().tolist())

filtered = df[
    df["precision"].isin(precision_filter) &
    df["arch"].isin(model_filter) &
    df["filename"].isin(dataset_filter)
]

# --- TABLE ---
display_cols = ["filename", "arch", "precision", "energy_consumed_kwh", "emissions_kg", "latency_seconds", "throughput_samples_per_sec", "accuracy", "created_at_str"]
display_cols = [c for c in display_cols if c in filtered.columns]
export_df = filtered[display_cols].rename(columns={"created_at_str": "created_at", "filename": "dataset", "arch": "model"}).reset_index(drop=True)
st.dataframe(export_df, use_container_width=True)

st.download_button(
    label="⬇️ Export to CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="experiments_export.csv",
    mime="text/csv",
)

st.divider()

# --- CHARTS ---
if filtered.empty:
    st.info("No data matches the current filters.")
    st.stop()

st.subheader("Visual Comparison")
st.caption("Bars are grouped by architecture; a batch that mixes MLP and CNN keeps them separate.")

_PREC_SCALE = alt.Scale(domain=["FP32", "INT8"], range=["#FF4B4B", "#00CC96"])


def _grouped_bar(col: str, y_title: str, zero_one: bool = False):
    d = filtered[["precision", "arch", col]].dropna()
    if d.empty:
        return
    y = alt.Y(f"mean({col}):Q", title=y_title,
             scale=alt.Scale(domain=[0, 1]) if zero_one else alt.Undefined)
    st.altair_chart(
        alt.Chart(d).mark_bar().encode(
            x=alt.X("arch:N", title=None),
            xOffset="precision:N",
            y=y,
            color=alt.Color("precision:N", scale=_PREC_SCALE),
            tooltip=["arch", "precision", alt.Tooltip(f"mean({col}):Q", format=".6g")],
        ).properties(height=300),
        use_container_width=True,
    )


chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### ⚡ Energy Consumed (kWh)")
    _grouped_bar("energy_consumed_kwh", "Avg kWh")
with chart_col2:
    st.markdown("#### ⏱ Latency (seconds)")
    _grouped_bar("latency_seconds", "Avg seconds")

chart_col2b, _ = st.columns(2)
with chart_col2b:
    st.markdown("#### 🚀 Throughput (samples/sec)")
    _grouped_bar("throughput_samples_per_sec", "Avg samples/sec")

chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    st.markdown("#### 🌍 Carbon Emissions (kg CO2)")
    _grouped_bar("emissions_kg", "Avg kg CO2")
with chart_col4:
    st.markdown("#### 🎯 Accuracy / R²")
    _grouped_bar("accuracy", "Avg accuracy or R²", zero_one=True)

st.divider()

# --- FP32 vs INT8 SAVINGS PER DATASET ---
st.subheader("FP32 vs INT8 Savings per Dataset")

# Keep only the latest run per dataset+precision to avoid duplicate index issues
latest = (
    filtered.sort_values("created_at")
    .groupby(["dataset_id", "precision"], as_index=False)
    .last()
)

fp32 = latest[latest["precision"] == "FP32"].set_index("dataset_id")
int8 = latest[latest["precision"] == "INT8"].set_index("dataset_id")
common = fp32.index.intersection(int8.index)

if common.empty:
    st.info("Need at least one dataset with both FP32 and INT8 runs to show savings.")
else:
    savings_rows = []
    for did in common:
        e32 = float(fp32.loc[did, "energy_consumed_kwh"] or 0)
        e8  = float(int8.loc[did, "energy_consumed_kwh"] or 0)
        l32 = float(fp32.loc[did, "latency_seconds"] or 0)
        l8  = float(int8.loc[did, "latency_seconds"] or 0)
        savings_rows.append({
            "dataset_id": did[:8] + "...",
            "energy_saved_kwh": round(e32 - e8, 8),
            "energy_saved_%": round((e32 - e8) / e32 * 100 if e32 > 0 else 0, 2),
            "latency_saved_sec": round(l32 - l8, 4),
            "latency_saved_%": round((l32 - l8) / l32 * 100 if l32 > 0 else 0, 2),
        })

    st.dataframe(pd.DataFrame(savings_rows), use_container_width=True)

st.divider()

# --- ACCURACY vs ENERGY TRADE-OFF SCATTER ---
st.subheader("🎯 Accuracy vs Energy Trade-off")
st.markdown(
    "Each point is one experiment run. The ideal model sits in the **bottom-right** corner: "
    "high accuracy, low energy."
)

scatter_df = filtered[["precision", "accuracy", "energy_consumed_kwh", "latency_seconds", "dataset_id"]].dropna()

if scatter_df.empty:
    st.info("Not enough data for scatter plot.")
else:
    scatter = (
        alt.Chart(scatter_df)
        .mark_circle(size=120, opacity=0.85)
        .encode(
            x=alt.X("energy_consumed_kwh:Q", title="Energy Consumed (kWh)"),
            y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "precision:N",
                scale=alt.Scale(domain=["FP32", "INT8"], range=["#FF4B4B", "#00CC96"]),
                legend=alt.Legend(title="Precision"),
            ),
            tooltip=[
                alt.Tooltip("precision:N", title="Precision"),
                alt.Tooltip("accuracy:Q", title="Accuracy", format=".4f"),
                alt.Tooltip("energy_consumed_kwh:Q", title="Energy (kWh)", format=".8f"),
                alt.Tooltip("latency_seconds:Q", title="Latency (s)", format=".3f"),
                alt.Tooltip("dataset_id:N", title="Dataset ID"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(scatter, use_container_width=True)

    # Annotation: which precision wins on each axis
    acc_fp32 = scatter_df[scatter_df["precision"] == "FP32"]["accuracy"].mean()
    acc_int8 = scatter_df[scatter_df["precision"] == "INT8"]["accuracy"].mean()
    e_fp32   = scatter_df[scatter_df["precision"] == "FP32"]["energy_consumed_kwh"].mean()
    e_int8   = scatter_df[scatter_df["precision"] == "INT8"]["energy_consumed_kwh"].mean()

    if not any(v != v for v in [acc_fp32, acc_int8, e_fp32, e_int8]):  # no NaN
        acc_loss  = round((acc_fp32 - acc_int8) * 100, 3)
        e_saving  = round((e_fp32 - e_int8) / e_fp32 * 100 if e_fp32 > 0 else 0, 1)
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Avg Accuracy Loss (FP32→INT8)", f"{acc_loss:+.3f}%")
        mcol2.metric("Avg Energy Saving (FP32→INT8)", f"{e_saving:.1f}%")
        worth = "Yes ✅" if abs(acc_loss) < 2 and e_saving > 10 else "Review ⚠️"
        mcol3.metric("INT8 Worth It?", worth)

st.divider()

# --- TRADE-OFF SCORE ---
st.subheader("⚖️ Trade-off Score")
st.markdown(
    "**Trade-off Score = Accuracy ÷ Energy (kWh)**  \n"
    "Measures how much accuracy you get per unit of energy consumed. "
    "A **higher score is better**. Use this to answer: *is INT8 worth the accuracy loss?*"
)

score_df = filtered[["precision", "accuracy", "energy_consumed_kwh", "dataset_id", "created_at"]].dropna()

if score_df.empty:
    st.info("Not enough data to compute trade-off scores.")
else:
    score_df = score_df.copy()
    score_df["trade_off_score"] = score_df["accuracy"] / score_df["energy_consumed_kwh"]

    # Summary metrics
    s_fp32 = score_df[score_df["precision"] == "FP32"]["trade_off_score"].mean()
    s_int8 = score_df[score_df["precision"] == "INT8"]["trade_off_score"].mean()

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Avg Score — FP32", f"{s_fp32:,.1f}" if s_fp32 == s_fp32 else "N/A")
    sc2.metric("Avg Score — INT8", f"{s_int8:,.1f}" if s_int8 == s_int8 else "N/A",
               delta=f"{((s_int8 - s_fp32) / s_fp32 * 100):+.1f}%" if s_fp32 > 0 else None)
    winner = "INT8" if s_int8 > s_fp32 else "FP32"
    sc3.metric("Better Efficiency", winner)

    st.markdown("#### Score per Experiment Run")

    # Bar chart of scores grouped by precision
    score_chart = (
        alt.Chart(score_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("precision:N", title=None),
            y=alt.Y("mean(trade_off_score):Q", title="Avg Accuracy / kWh"),
            color=alt.Color(
                "precision:N",
                scale=alt.Scale(domain=["FP32", "INT8"], range=["#FF4B4B", "#00CC96"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("precision:N", title="Precision"),
                alt.Tooltip("mean(trade_off_score):Q", title="Avg Score", format=",.1f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(score_chart, use_container_width=True)

    # Per-dataset breakdown table
    st.markdown("#### Per-dataset Breakdown")
    latest_scores = (
        score_df.sort_values("created_at")
        .groupby(["dataset_id", "precision"], as_index=False)
        .last()[["dataset_id", "precision", "accuracy", "energy_consumed_kwh", "trade_off_score"]]
    )
    latest_scores["dataset_id"] = latest_scores["dataset_id"].str[:8] + "..."
    latest_scores = latest_scores.rename(columns={
        "dataset_id": "Dataset",
        "precision": "Precision",
        "accuracy": "Accuracy",
        "energy_consumed_kwh": "Energy (kWh)",
        "trade_off_score": "Trade-off Score",
    })
    st.dataframe(
        latest_scores.sort_values("Trade-off Score", ascending=False).reset_index(drop=True),
        use_container_width=True
    )

st.divider()

# --- RADAR CHART ---
st.subheader("🕸️ Multi-dimensional Comparison (Radar)")
st.markdown(
    "All metrics normalized to 0–1. **Larger area = better overall profile.**  \n"
    "Accuracy and Throughput: higher is better. Energy and Emissions: lower is better (inverted)."
)

radar_df = filtered[["precision", "accuracy", "energy_consumed_kwh",
                      "emissions_kg", "latency_seconds",
                      "throughput_samples_per_sec"]].dropna()

if radar_df.empty:
    st.info("Not enough data for radar chart.")
else:
    avg = radar_df.groupby("precision").mean(numeric_only=True)

    dims = {
        "Accuracy":     ("accuracy",                   False),
        "Throughput":   ("throughput_samples_per_sec", False),
        "Low Energy":   ("energy_consumed_kwh",        True),
        "Low Emissions":("emissions_kg",               True),
        "Low Latency":  ("latency_seconds",            True),
    }

    def normalize(series, invert):
        mn, mx = series.min(), series.max()
        normed = (series - mn) / (mx - mn) if mx > mn else series * 0 + 0.5
        return 1 - normed if invert else normed

    labels = list(dims.keys())
    fig = go.Figure()
    colors = {"FP32": "#FF4B4B", "INT8": "#00CC96"}

    for precision in avg.index:
        values = []
        for label, (col, invert) in dims.items():
            col_series = avg[col]
            val = normalize(col_series, invert).get(precision, 0.5)
            values.append(round(float(val), 4))
        values += values[:1]  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=precision,
            line_color=colors.get(precision, "#888"),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)
