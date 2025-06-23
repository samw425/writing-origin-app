import streamlit as st
import joblib
import numpy as np

# Load existing vectorizer and model (no changes here)
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def explain_prediction(text):
    vec = vectorizer.transform([text])
    coef = model.coef_
    features = vectorizer.get_feature_names_out()
    pred_class = model.predict(vec)[0]
    class_idx = list(model.classes_).index(pred_class)
    feature_contributions = coef[class_idx] * vec.toarray()[0]

    # Sort by absolute contribution, descending
    sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]

    explanations = []
    count = 0
    generic_ngrams = ['the', 'and', 'ing', 'ion', 'to', 'in', 'of', 'is']

    for i in sorted_indices:
        weight = feature_contributions[i]
        if abs(weight) < 0.01:
            # Skip very low impact features
            continue
        
        feat = features[i]
        
        # Skip very generic/common features that add no info
        if any(g in feat for g in generic_ngrams):
            continue
        
        # Map some frequent ngrams to human-friendly explanations
        if "n't" in feat:
            explanations.append("use of negative contractions like 'n't'")
        elif "'re" in feat:
            explanations.append("use of contractions like 'you're'")
        elif 'll' in feat:
            explanations.append("use of future tense contractions like 'I'll'")
        elif ' y' in feat or "ya" in feat or "yo" in feat:
            explanations.append("use of informal/slang terms")
        elif ' u' in feat or 'you' in feat:
            explanations.append("use of direct address pronouns")
        else:
            explanations.append(f"usage of the pattern '{feat}'")
        
        count += 1
        if count >= 5:
            break

    if not explanations:
        return f"The text is predicted to be from {pred_class}, but no distinctive writing features were found to explain this clearly."

    return f"The text shows features common to writing from {pred_class}, including {', '.join(explanations)}."

if st.button("Predict Origin"):
    if user_input.strip():
        prediction = model.predict(vectorizer.transform([user_input]))[0]
        explanation = explain_prediction(user_input)
        st.success(f"Predicted Origin: {prediction}")
        st.write(explanation)
    else:
        st.error("Please enter some text to analyze.")
