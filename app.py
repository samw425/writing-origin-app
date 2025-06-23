import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def explain_prediction(input_text, top_features=5):
    # Vectorize input
    input_vec = vectorizer.transform([input_text])
    
    # Predict probabilities
    probs = model.predict_proba(input_vec)[0]
    classes = model.classes_
    
    # Sort predictions by confidence
    sorted_indices = np.argsort(probs)[::-1]
    top_index = sorted_indices[0]
    second_index = sorted_indices[1]
    
    top_pred = (classes[top_index], probs[top_index]*100)
    second_pred = (classes[second_index], probs[second_index]*100)
    
    # Feature explanation
    input_ngrams = input_vec.nonzero()[1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_class_idx = list(classes).index(top_pred[0])
    top_coef = model.coef_[top_class_idx]
    
    top_features_idx = input_ngrams
    top_weights = top_coef[top_features_idx]
    
    top_n_idx = np.argsort(top_weights)[::-1][:top_features]
    contributing_patterns = feature_names[top_features_idx[top_n_idx]]
    
    return top_pred, second_pred, contributing_patterns

def build_explanation(main, backup, patterns):
    pred_region, pred_conf = main
    backup_region, backup_conf = backup
    
    explanation = f"**Predicted Origin: {pred_region}**\n\n"
    
    if pred_conf > backup_conf * 1.5:
        explanation += "The model is confident in this prediction.\n\n"
    else:
        explanation += f"The model leans toward this prediction, but there are writing features also present in {backup_region}.\n\n"
    
    if len(patterns) > 0:
        feature_phrase = ", ".join([f"'{p}'" for p in patterns])
        explanation += f"This is based on stylistic features such as: {feature_phrase}."
    else:
        explanation += "However, no distinctive writing features were strong enough to explain this clearly."
    
    return explanation

if st.button("Predict Origin"):
    if user_input.strip():
        top_pred, second_pred, patterns = explain_prediction(user_input)
        explanation = build_explanation(top_pred, second_pred, patterns)
        st.markdown(explanation)
    else:
        st.error("Please enter some text to analyze.")

