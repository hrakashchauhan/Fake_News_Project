# Import all the necessary libraries
import streamlit as st
import joblib
import re
import nltk # Keep this import
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# --- 1. Load the Saved Models ---
# We are loading the trained model and the TF-IDF vectorizer that we saved earlier.
# This ensures we don't have to retrain the model every time we run the app.
try:
    vectorizer = joblib.load('tfidf_vectorizer_v2.pkl')
model = joblib.load('svm_model_v2.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please make sure 'tfidf_vectorizer.pkl' and 'svm_model.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

# --- 2. Re-create the Text Cleaning Function ---
# This is the exact same function we used in our notebook to clean the training data.
# Any new text from the user MUST go through the same cleaning process.
port_stem = PorterStemmer()

def stem_text(content):
    # The [^a-zA-Z] pattern means "anything that is NOT a letter".
    # We replace all non-letters with a space.
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # Stemming reduces words to their root form (e.g., "running" becomes "run").
    # We also remove common English "stopwords" (like 'the', 'a', 'is').
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# --- 3. Create the Streamlit User Interface ---
st.title("Fake News Detector 📰")
st.write(
    "Enter the text of a news article below to check if it is likely Real or Fake."
)

# Create a text area for user input
input_text = st.text_area("Enter news article text here:", height=250)

# Create a button to trigger the prediction
if st.button("Detect News"):
    if input_text:
        # --- 4. The Prediction Pipeline ---
        # When the button is clicked, we follow these steps:

        # Step 1: Clean the user's input text.
        cleaned_text = stem_text(input_text)

        # Step 2: Convert the cleaned text into numerical features using the loaded vectorizer.
        # Note: We use `.transform()` here, not `.fit_transform()`, because the vectorizer is already trained.
        vectorized_text = vectorizer.transform([cleaned_text])

        # Step 3: Use the loaded model to make a prediction.
        prediction = model.predict(vectorized_text)

        # --- 5. Display the Result ---
        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.success("✅ This looks like REAL news.")
        else:
            st.error("❌ This looks like FAKE news.")
    else:
        st.warning("Please enter some text to analyze.")
