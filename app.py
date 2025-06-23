import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

def explain_prediction(text, model, vectorizer, top_n=5):
    vec = vectorizer.transform([text])
    coefs = model.coef_
    classes = model.classes_
    prediction = model.predict(vec)[0]
    class_idx = list(classes).index(prediction)

    feature_names = vectorizer.get_feature_names_out()
    text_vec = vec.toarray()[0]
    contributions = coefs[class_idx] * text_vec

    # Get top n contributing n-grams with positive weights
    top_indices = np.argsort(contributions)[-top_n:][::-1]
    top_features = [(feature_names[i], contributions[i]) for i in top_indices if contributions[i] > 0]

    explanation = f"Predicted origin: {prediction}\n\nTop contributing character n-grams:\n"
    for feat, val in top_features:
        explanation += f" - '{feat}': weight {val:.4f}\n"

    return explanation

st.title("Writing Origin Predictor with Explainability")

user_input = st.text_area("Enter a sample of your writing:")

if st.button("Analyze"):
    if user_input.strip():
        # Vectorize input and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Origin: {prediction}")

        explanation = explain_prediction(user_input, model, vectorizer)
        st.text_area("Why this prediction?", explanation, height=200)
    else:
        st.error("Please enter some text to analyze.")
