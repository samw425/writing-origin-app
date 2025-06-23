import streamlit as st
import joblib

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

if st.button("Predict Origin"):
    if user_input.strip():
        # Vectorize input and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Origin: {prediction}")
    else:
        st.error("Please enter some text to analyze.")
