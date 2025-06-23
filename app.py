import streamlit as st
import joblib
import numpy as np
import re

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def is_latin(text):
    """Check if the text is composed of only Latin characters"""
    # Returns True if all chars are in ASCII range (basic Latin)
    return all(ord(c) < 128 for c in text)

def get_readable_features(top_features, predicted_origin):
    """
    Convert top n-grams to a user-friendly explanation.
    For non-Latin scripts, use plain-language phrases.
    """
    # Detect if majority of top features are non-latin
    non_latin_count = sum(not is_latin(f) for f in top_features)
    total = len(top_features)
    
    # If most features are non-latin, map to generic descriptions
    if non_latin_count > total / 2:
        # Map some known origins to descriptions
        mapping = {
            'China': "common Chinese character patterns typical in Mandarin writing",
            'Russia': "common Cyrillic character patterns typical in Russian writing",
            'India': "common Indic script patterns or English influenced by Indian syntax",
            # Add more as needed
        }
        description = mapping.get(predicted_origin, "unique non-Latin writing style patterns")
        return [description]
    else:
        # For Latin script, return features quoted
        return [f"'{f}'" for f in top_features]

def generate_explanation(text, vectorizer, model):
    input_vec = vectorizer.transform([text])
    
    # Get prediction and prediction probs
    pred = model.predict(input_vec)[0]
    probs = model.predict_proba(input_vec)[0]
    
    # Sort indices by confidence descending
    sorted_indices = np.argsort(probs)[::-1]
    
    top_idx = sorted_indices[0]
    top_prob = probs[top_idx]
    top_origin = model.classes_[top_idx]
    
    # Second best
    if len(probs) > 1:
        second_idx = sorted_indices[1]
        second_prob = probs[second_idx]
        second_origin = model.classes_[second_idx]
    else:
        second_prob = None
        second_origin = None
    
    # Get feature log odds for top class
    if hasattr(model, 'coef_'):
        coefs = model.coef_[top_idx]
        feature_names = vectorizer.get_feature_names_out()
        
        # Transform input text vector into array
        input_array = input_vec.toarray()[0]
        
        # Element-wise multiply to get feature contributions present in input
        contributions = coefs * input_array
        
        # Get indices of positive contributions sorted descending
        top_feature_indices = contributions.argsort()[::-1]
        
        # Take top 5 features with positive contributions
        top_features = []
        for idx in top_feature_indices:
            if contributions[idx] > 0 and input_array[idx] > 0:
                top_features.append(feature_names[idx])
            if len(top_features) >= 5:
                break
    else:
        top_features = []
    
    readable_features = get_readable_features(top_features, top_origin)
    
    # Build explanation text
    explanation_lines = []
    explanation_lines.append(f"Predicted Origin: **{top_origin}**")
    explanation_lines.append(f"The model is confident in this prediction.")
    explanation_lines.append("")
    explanation_lines.append("This is based on writing style features such as: " + ", ".join(readable_features) + ".")
    
    # Include secondary origin only if confidence is reasonably close (e.g. > 10%) and different
    if second_origin and second_origin != top_origin and second_prob > 0.1:
        explanation_lines.append("")
        explanation_lines.append(f"Possible secondary origin: **{second_origin}** — due to some overlapping writing features.")
    
    return "\n\n".join(explanation_lines)

if st.button("Predict Origin"):
    if user_input.strip():
        explanation = generate_explanation(user_input, vectorizer, model)
        st.markdown(explanation)
    else:
        st.error("Please enter some text to analyze.")
