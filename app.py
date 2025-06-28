# ===================================================================
# THIS IS YOUR UPDATED app.py FILE
# ===================================================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from huggingface_hub import hf_hub_download # New Import

# --- NLTK Data Download (Keep the robust version) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# ----------------------------------------

# --- 1. Load the Saved Models from Hugging Face ---
# This function downloads the model from your Hugging Face Hub repository
# and caches it so it doesn't re-download on every run.
@st.cache_resource
def load_model_from_hf():
    try:
        # !!! IMPORTANT: Replace 'hrakashchauhan' with your actual Hugging Face username !!!
        repo_id = "hrakashchauhan/satya-checker-svm-v1" # Or whatever you named your repo
        
        model_path = hf_hub_download(repo_id=repo_id, filename="svm_model_v2.pkl")
        vectorizer_path = hf_hub_download(repo_id=repo_id, filename="tfidf_vectorizer_v2.pkl")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models from Hugging Face Hub: {e}")
        return None, None

# Load the models when the app starts
model, vectorizer = load_model_from_hf()

# If models fail to load, stop the app gracefully
if model is None or vectorizer is None:
    st.warning("Models could not be loaded. The app cannot proceed.")
    st.stop()

# --- 2. Re-create the Text Cleaning Function ---
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_text(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    return ' '.join(stemmed_content)

# --- 3. Create the Streamlit User Interface ---
st.title("Satya-Checker (सत्य-चेकर) 📰")
st.header("The Universal Misinformation Shield")
st.write("Enter the text of any news article below to check its credibility.")

input_text = st.text_area("Enter news article text here:", height=250)

if st.button("Analyze Credibility"):
    if input_text:
        # Prediction Pipeline
        cleaned_text = stem_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)

        # Display Result
        st.subheader("Credibility Analysis:")
        if prediction[0] == 1:
            st.success("✅ Analysis suggests this content is LIKELY CREDIBLE.")
        else:
            st.error("❌ Analysis suggests this content is LIKELY MISINFORMATION.")
    else:
        st.warning("Please enter some text to analyze.")

# ===================================================================
# THIS IS YOUR UPDATED requirements.txt FILE
# ===================================================================
# streamlit
# scikit-learn
# pandas
# nltk
# joblib
# swifter
# huggingface-hub  <- ADD THIS NEW LINE
