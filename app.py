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

    top_indices = np.argsort(contributions)[-top_n:][::-1]
    top_features = [(feature_names[i], contributions[i]) for i in top_indices if contributions[i] > 0]

    # Map common n-grams to plain English explanations
    explanation_map = {
        "we": "use of inclusive pronouns like 'we'",
        "n't": "frequent use of contractions (e.g., 'don't', 'can't')",
        "'t": "common negation patterns",
        "in": "typical use of prepositions",
        "the": "usage of the definite article",
        "you": "direct address to the reader",
        "g'day": "informal greeting common in Australian English",
        "mate": "friendly term often used in British and Australian English",
        "y'all": "colloquial plural you common in Southern US English",
        "barbecue": "cultural reference typical in Southern US",
        # Add more mappings as needed for your data
    }

    explanations = []
    for feat, val in top_features:
        # Clean n-gram from spaces and apostrophes for matching keys
        clean_feat = feat.strip().replace("'", "").lower()
        plain = explanation_map.get(clean_feat, f"usage of the pattern '{feat.strip()}'")
        explanations.append(plain)

    explanation_text = (
        f"The text shows features common to writing from {prediction}, such as "
        + ", ".join(explanations)
        + "."
    )
    return explanation_text

st.title("Writing Origin Predictor with Plain English Explanation")

user_input = st.text_area("Enter a sample of your writing:")

if st.button("Analyze"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Origin: {prediction}")

        explanation = explain_prediction(user_input, model, vectorizer)
        st.text_area("Why this prediction?", explanation, height=150)
    else:
        st.error("Please enter some text to analyze.")

