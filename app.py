import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def generate_explanation(coef, feature_names, top_n=5):
    # Get indices of top positive features for the predicted class
    top_indices = np.argsort(coef)[-top_n:]
    top_features = [feature_names[i] for i in reversed(top_indices)]
    return top_features

if st.button("Predict Origin"):
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        # Vectorize input and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0]

        classes = model.classes_
        predicted_index = list(classes).index(prediction)

        # Find secondary origin if close confidence (within 8% of main)
        sorted_indices = np.argsort(probs)[::-1]
        primary_idx = sorted_indices[0]
        secondary_idx = sorted_indices[1]

        primary_confidence = probs[primary_idx]
        secondary_confidence = probs[secondary_idx]

        # Prepare explanation features for predicted class
        coef = model.coef_[primary_idx]
        feature_names = vectorizer.get_feature_names_out()
        top_features = generate_explanation(coef, feature_names, top_n=6)

        # Build explanation text
        explanation = (
            f"The writing style shows key features such as: {', '.join(top_features)}."
        )

        st.markdown(f"### Predicted Origin: {prediction}")

        if secondary_confidence >= primary_confidence - 0.08:
            secondary_origin = classes[secondary_idx]
            st.markdown(
                f"**Possible secondary origin:** {secondary_origin} — "
                "due to overlapping writing features."
            )
        else:
            secondary_origin = None

        st.markdown("### Why this prediction?")
        st.write(explanation)
