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
    
    # Analyze n-grams
    input_ngrams = vectorizer.transform([input_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[list(classes).index(main_prediction)]
    top_features_idx = input_ngrams.nonzero()[1]
    top_features_weights = coef[top_features_idx]
    
    top_n_idx = np.argsort(top_features_weights)[::-1][:top_features]
    contributing_patterns = feature_names[top_features_idx[top_n_idx]]
    
    return main_prediction, top_preds, contributing_patterns

if st.button("Predict Origin"):
    if user_input.strip():
        prediction, top_preds, patterns = explain_prediction(user_input)

        st.success(f"Predicted Origin: {prediction}")

        # Show Top 3 likely predictions
        st.markdown("**Top 3 Prediction Confidence Scores:**")
        for region, score in top_preds:
            st.markdown(f"- **{region}**: {score:.2f}%")

        # Explanation
        st.markdown("**Why this prediction?**")
        if patterns.size > 0:
            pattern_text = ", ".join([f"'{p}'" for p in patterns])
            st.write(f"The text is predicted to be from {prediction}. This is because it shows writing features such as: {pattern_text}.")
        else:
            st.write(f"The text is predicted to be from {prediction}, but no distinctive writing patterns were detected.")
    else:
        st.error("Please enter some text to analyze.")

