# ===================================================================================
# "SATYA-CHECKER" APP V3.0 - POWERED BY A UNIFIED PIPELINE
# This version is simpler and more robust, preventing training/prediction mismatches.
# ===================================================================================

import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

# --- 1. Load the Unified Pipeline from Hugging Face ---
# This function downloads the single pipeline file that contains the
# text cleaner, vectorizer, and model all bundled together.
@st.cache_resource
def load_pipeline():
    try:
        # !!! IMPORTANT: Make sure this matches your Hugging Face username and repo name !!!
        repo_id = "hrakashchauhan/satya-checker-svm-v1" 
        pipeline_filename = "asli_nakli_pipeline_v1.pkl"
        
        # Download the pipeline file from the Hub
        pipeline_path = hf_hub_download(repo_id=repo_id, filename=pipeline_filename)
        
        # Load the pipeline from the downloaded file
        pipeline = joblib.load(pipeline_path)
        return pipeline
        
    except Exception as e:
        st.error(f"Error loading the model pipeline from Hugging Face Hub: {e}")
        return None

# Load the pipeline when the app starts
pipeline = load_pipeline()

# If the pipeline fails to load, stop the app gracefully
if pipeline is None:
    st.warning("The AI model pipeline could not be loaded. The app cannot proceed.")
    st.stop()


# --- 2. Create the Streamlit User Interface ---
st.title("Asli-Nakli News Checker 📰")
st.write(
    "Enter the text of any news article below to check if it is likely Asli (Real) or Nakli (Fake)."
)

# Create a text area for the user to paste news content
input_text = st.text_area("Enter news article text here:", height=250)

# Create a button to trigger the analysis
if st.button("Analyze Credibility"):
    if input_text:
        # --- 3. The Prediction Pipeline (Now Much Simpler) ---
        
        # The pipeline handles everything automatically in one step:
        # 1. It takes the raw user text.
        # 2. It applies the exact same text cleaning from training.
        # 3. It applies the exact same TF-IDF vectorization from training.
        # 4. It makes a prediction using the trained classifier.
        prediction = pipeline.predict([input_text])

        # --- 4. Display the Result ---
        st.subheader("Credibility Analysis:")
        # The output is an array (e.g., [1]), so we check the first element.
        if prediction[0] == 1:
            st.success("✅ Analysis suggests this content is LIKELY REAL.")
        else:
            st.error("❌ Analysis suggests this content is LIKELY FAKE.")
    else:
        st.warning("Please enter some text to analyze.")

