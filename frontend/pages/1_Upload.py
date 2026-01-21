import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Upload Dataset")

st.header("📂 Upload New Dataset")
st.markdown("""
    Here you can upload `.csv` files to be processed by the backend.
    Please ensure your dataset format matches the model you select (e.g., flattened pixels for CNN).
""")
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Current Backend:")
    st.code(API_URL, language="text")


st.divider()

# --- UPLOAD FORM ---
with st.form("upload_form", clear_on_submit=True):
    # 1. File Input
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # 2. Model Selection (Matches your Backend Enum)
    # We use specific values "MLP" and "CNN" to match the backend expectations
    ai_model_type = st.selectbox(
        "Select Target Model Architecture", 
        ["MLP", "CNN"],
        help="Select 'MLP' for tabular/sensor data or 'CNN' for image pixel data (MNIST)."
    )
    
    # 3. Description
    description = st.text_area("Dataset Description", placeholder="e.g., MNIST Mini test set with 100 rows...")
    
    # 4. Submit Button
    submitted = st.form_submit_button("📂 Upload Dataset")

    if submitted:
        if uploaded_file is not None:
            # Prepare the payload
            # 'file' matches the @UploadFile parameter in FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            
            # 'params' matches the query parameters in FastAPI
            params = {
                "ai_model": ai_model_type, 
                "description": description
            }
            
            with st.spinner("Uploading to Backend..."):
                try:
                    # 
                    response = requests.post(f"{API_URL}/datasets", files=files, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Upload Successful!")
                        st.json({
                            "ID": data['id'],
                            "Filename": data['filename'],
                            "Model": data['ai_model']
                        })
                        st.info("You can now go to the **Experiments** page to run tests on this file.")
                        st.page_link("pages/2_Experiments.py", label="Start Experiments", icon="🧪")
                    else:
                        st.error(f"❌ Upload Failed: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the Backend. Is it running?")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")
        else:
            st.warning("⚠️ Please select a file first.")