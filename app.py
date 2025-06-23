import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def get_feature_explanations(input_vec, model, vectorizer, top_n=5):
    """
    Extract top character n-grams contributing to prediction for explanation.
    Returns list of human-friendly patterns.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_
    classes = model.classes_

    # Predict probabilities and get top class index
    pred_probs = model.predict_proba(input_vec)[0]
    pred_class_idx = np.argmax(pred_probs)
    
    # Get the coefficient vector for predicted class
    class_coefs = coefs[pred_class_idx]

    # Find non-zero features in input vector
    input_indices = input_vec.indices

    # Get the coefficients for features present in input
    present_coefs = class_coefs[input_indices]

    # Pair feature names with coef weights
    feature_contribs = list(zip(feature_names[input_indices], present_coefs))

    # Sort by absolute contribution descending
    feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Take top_n and format nicely
    top_features = [f"'{feat}'" for feat, weight in feature_contribs[:top_n]]

    return top_features

if st.button("Predict Origin"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0]
        pred_probs = model.predict_proba(input_vec)[0]

        # Sort predictions by probability descending
        sorted_indices = np.argsort(pred_probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]

        top_origin = model.classes_[top_idx]
        top_confidence = pred_probs[top_idx] * 100
        second_origin = model.classes_[second_idx]
        second_confidence = pred_probs[second_idx] * 100

        # Get feature explanations for top prediction
        explanations = get_feature_explanations(input_vec, model, vectorizer, top_n=5)

        st.success(f"Predicted Origin: {top_origin}")
        st.markdown(f"**Confidence:** {top_confidence:.2f}%")

        explanation_text = (
            f"The model is confident in this prediction based on stylistic features such as: "
            + ", ".join(explanations)
            + "."
        )
        st.write(explanation_text)

        # Show backup prediction only if confidence is reasonably close (say within 10%)
        confidence_gap = top_confidence - second_confidence
        if confidence_gap < 10:
            st.markdown(f"---")
            st.write(
                f"**Possible secondary origin:** {second_origin} ({second_confidence:.2f}%) "
                f"— due to some overlapping writing features."
            )
    else:
        st.error("Please enter some text to analyze.")

        st.error("Please enter some text to analyze.")
