import streamlit as st
import altair as alt
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load Config
load_dotenv()
API_URL = os.getenv("BACKEND_URL")

st.set_page_config(page_title="Run Experiments", layout="wide")

st.title("⚡ Experiment Runner")
st.markdown("Compare the **Energy Consumption** between FP32 and INT8 models.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Current Backend:")
    st.code(API_URL, language="text")
    n_runs = st.slider(
        "Repeat runs (averaged)",
        min_value=1, max_value=10, value=1,
        help="Run each precision N times and average the results for more reliable measurements."
    )
    if st.button("🔄 Refresh Datasets"):
        st.rerun()

# --- HELPER FUNCTION: PLOT CHART ---
def display_charts(fp32_data, int8_data):
    """
    Renders stacked charts for Energy and Emissions.
    """
    st.subheader("📊 Visual Analysis")

    # 1. Extract Data
    e_fp32 = fp32_data.get('energy_consumed_kwh', 0)
    e_int8 = int8_data.get('energy_consumed_kwh', 0)
    
    c_fp32 = fp32_data.get('emissions_kg', 0)
    c_int8 = int8_data.get('emissions_kg', 0)

    # --- CHART 1: ENERGY (kWh) ---
    st.markdown("#### ⚡ Energy Consumption (kWh)")
    df_energy = pd.DataFrame({
        "Precision": ["FP32", "INT8"],
        "Value": [e_fp32, e_int8]
    })
    
    chart_energy = alt.Chart(df_energy).mark_bar().encode(
        x=alt.X('Precision', title=None),
        y=alt.Y('Value', title='kWh'),
        color=alt.Color('Precision', 
                        scale=alt.Scale(range=['#FF4B4B', '#00CC96']), # Red vs Green
                        legend=None),
        tooltip=['Precision', 'Value']
    ).properties(
        height=300 # Slightly shorter since it's full width now
    )
    
    st.altair_chart(chart_energy, use_container_width=True)

    st.divider() # clean separation line

    # --- CHART 2: EMISSIONS (kgCO2eq) ---
    st.markdown("#### 🌍 Carbon Emissions (kg)")
    df_emissions = pd.DataFrame({
        "Precision": ["FP32", "INT8"],
        "Value": [c_fp32, c_int8]
    })
    
    chart_emissions = alt.Chart(df_emissions).mark_bar().encode(
        x=alt.X('Precision', title=None),
        y=alt.Y('Value', title='kg CO2'),
        color=alt.Color('Precision', 
                        scale=alt.Scale(range=['#FF4B4B', '#00CC96']), 
                        legend=None),
        tooltip=['Precision', 'Value']
    ).properties(
        height=300
    )
    
    st.altair_chart(chart_emissions, use_container_width=True)


# --- STEP 1: SELECT DATASET ---
st.subheader("1. Select a Dataset")

try:
    response = requests.get(f"{API_URL}/datasets")
    if response.status_code == 200:
        datasets = response.json().get("datasets", [])
    else:
        st.error(f"Backend Error: {response.status_code}")
        datasets = []
except Exception as e:
    st.error(f"Connection Failed: {e}")
    datasets = []

if datasets:
    options = {f"{d['filename']} ({d['ai_model']})": d['id'] for d in datasets}
    selected_label = st.selectbox("Choose dataset:", list(options.keys()))
    selected_id = options[selected_label]
    
    selected_data = next(d for d in datasets if d['id'] == selected_id)
    st.info(f"**Model:** {selected_data['ai_model']} | **Uploaded:** {selected_data['created_at'][:10]}")

    st.divider()

    # --- STEP 2: CHECK FOR EXISTING HISTORY ---
    history_found = False
    
    # Try to fetch history using your NEW endpoint
    try:
        hist_resp = requests.get(f"{API_URL}/experiments/{selected_id}")
        if hist_resp.status_code == 200:
            hist_data = hist_resp.json()
            st.success("📅 **Found existing results** for this dataset:")
            
            # Show the chart immediately using existing data
            display_charts(hist_data['fp32'], hist_data['int8'])
            history_found = True
            
            st.divider()
    except Exception:
        pass # If fails, we just show the "Run" button normally

    # --- STEP 3: RUN NEW EXPERIMENT ---
    st.subheader("2. Run Comparison" if not history_found else "3. Re-Run Comparison")
    
    btn_label = "🚀 Start Comparison Experiment" if not history_found else "🔄 Run New Comparison"
    
    if st.button(btn_label, type="primary"):
        
        progress_text = "Operation in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            run_label = f"{n_runs} run(s)" if n_runs > 1 else "1 run"
            my_bar.progress(10, text=f"Initializing Models ({run_label})...")

            resp = requests.get(f"{API_URL}/compare/{selected_id}", params={"n_runs": n_runs})

            my_bar.progress(80, text="Processing Results...")

            if resp.status_code == 200:
                data = resp.json()

                fp32_res = data.get('fp32_results') or data.get('fp32')
                int8_res = data.get('int8_results') or data.get('int8')
                averaged = data.get('averaged', {})
                improvement = data.get('improvement', {})

                my_bar.progress(100, text="Done!")
                time.sleep(0.5)
                my_bar.empty()

                st.success(f"✅ Experiment Completed ({run_label})!")

                if n_runs > 1:
                    st.markdown(f"#### 📐 Averaged Results ({n_runs} runs)")
                    ac1, ac2, ac3 = st.columns(3)
                    ac1.metric("Avg Energy FP32 (kWh)", f"{averaged.get('fp32_energy_kwh', 0):.8f}")
                    ac2.metric("Avg Energy INT8 (kWh)", f"{averaged.get('int8_energy_kwh', 0):.8f}",
                               delta=f"{improvement.get('energy_saved_percentage', 0):+.1f}%")
                    ac3.metric("Accuracy Loss", f"{improvement.get('accuracy_loss', 0):+.4f}")

                display_charts(fp32_res, int8_res)
                
                # Optional: Rerun to update the "History" view automatically
                # st.rerun() 

            else:
                my_bar.empty()
                st.error(f"Experiment Failed: {resp.text}")

        except Exception as e:
            my_bar.empty()
            st.error(f"Error connecting to backend: {e}")

else:
    st.warning("⚠️ No datasets found. Please go to the 'Upload' page first.")