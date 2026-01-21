import os
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.header("Energy-Aware Logging Mechanism")
st.subheader("Monitor and Analyze Energy Consumption Data")

API_URL = os.getenv("BACKEND_URL")


st.divider()

st.subheader("🚀 What would you like to do?")

action_col1, action_col2 = st.columns(2)

with action_col1:
    st.info("🆕 **New Data?**")
    st.markdown("Upload a new CSV file.")
    # Streamlit Page Navigation (if using pages folder)
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📂")

with action_col2:
    st.success("⚡ **Ready to Test?**")
    st.markdown("Run FP32 vs INT8 comparison on existing data.")
    st.page_link("pages/2_Experiments.py", label="Start Experiments", icon="🧪")
    

try:
    datasets_resp = requests.get(f"{API_URL}/datasets")
    if datasets_resp.status_code == 200:
        datasets = datasets_resp.json().get("datasets", [])
        dataset_count = len(datasets)
    else:
        dataset_count = 0
except Exception as e:
    dataset_count = f"Error: {e}"

st.divider()

st.subheader("📂 Recent Datasets")
if isinstance(dataset_count, int) and dataset_count > 0:
    # Convert list of dicts to DataFrame for a pretty table
    df = pd.DataFrame(datasets)
    # Select only useful columns to show
    if not df.empty:
        display_df = df[['description','filename', 'ai_model', 'created_at', 'id']]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.warning("No datasets found. Please upload one!")