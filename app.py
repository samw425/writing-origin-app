import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def explain_prediction(input_text, top_features=5):
    # Vectorize the input
    input_vec = vectorizer.transform([input_text])
    
    # Predict probabilities
    probs = model.predict_proba(input_vec)[0]
    classes = model.classes_
    
    # Get top 3 predictions
    top_indices = np.argsort(probs)[::-1][:3]
    top_preds = [(classes[i], probs[i]*100) for i in top_indices]
    
    # Get main prediction
    main_prediction = top_preds[0][0]
    
    # Get n-gram features contributing to that class
    feature_names = np.array(vectorizer.get_feature_names_out())
    input_indices = input_vec.nonzero()[1]
    coef = model.coef_[list(classes).index(main_prediction)]
    weights = coef[input_indices]
    
    # Get top character patterns
    top_n_idx = np.argsort(weights)[::-1][:top_features]
    top_patterns = feature_names[input_indices[top_n_idx]]
    
    # Convert patterns into readable interpretation
    natural_phrases = []
    for p in top_patterns:
        if len(p.strip()) < 2:
            continue
        elif "n't" in p or "'t" in p:
            natural_phrases.append("use of contractions like 'don't' or 'isn't'")
        elif p.startswith("th"):
            natural_phrases.append("frequent use of 'th' patterns, common in English articles and pronouns")
        elif p.startswith("is ") or p.startswith(" is"):
            natural_phrases.append("structured sentence openings like 'It is', 'This is'")
        elif p.endswith(" a"):
            natural_phrases.append("phrase endings like 'is a' or 'was a'")
        elif p.strip().isalpha() and len(p) > 3:
            natural_phrases.append(f"use of the pattern '{p}'")
    
    if not natural_phrases:
        explanation = f"The text is predicted to be from {main_prediction}, but no clear stylistic cues were strongly detected."
    else:
        explanation = f"The text is predicted to be from {main_prediction}. This is based on stylistic features such as: " + ", ".join(natural_phrases[:top_features]) + "."

    return main_prediction, top_preds, explanation

if st.button("Predict Origin"):
    if user_input.strip():
        prediction, top_preds, explanation = explain_prediction(user_input)

        st.success(f"Predicted Origin: {prediction}")

        # Show Top 3 predictions
        st.markdown("**Top 3 Prediction Confidence Scores:**")
        for region, score in top_preds:
            st.markdown(f"- **{region}**: {score:.2f}%")

        # Show human-style explanation
        st.markdown("**Why this prediction?**")
        st.write(explanation)

    else:
        st.error("Please enter some text to analyze.")


