import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#ps = PorterStemmer()
# Download stopwords if not already present
nltk.download("stopwords")
#nltk.download('punkt')

# Load the vectorizer and model
tfidf1 = pickle.load(open('vectorizer.pkl', 'rb'))
svm_model = pickle.load(open('model.pkl', 'rb'))

# Initialize stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Define text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()  # Tokenization (split into words)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    return ' '.join(words)  # Rejoin words into processed text

# Streamlit UI
st.title("Spam E-Mail Classifier")
st.write("Enter your email text here and the model will predict whether it's spam or not.")

# Use text_area instead of text_input for better formatting with long emails
input_message = st.text_area("Enter the message:", "")

# Prediction Button
if st.button("Predict"):
    if input_message.strip():  # Ensure input is not empty
        # Preprocess
        transformed_message = transform_text(input_message)

        # Vectorize
        vectorized_message = tfidf1.transform([transformed_message])

        # Predict
        prediction = svm_model.predict(vectorized_message)[0]

        # Display Result
        if prediction == 1:
            st.error("This is a Spam E-mail.")
        else:
            st.success("This is Not a Spam E-mail.")
    else:
        st.warning("Please enter a message before predicting.")
