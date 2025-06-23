import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

def explain_features(text, vectorizer, model, top_n=5):
    vec = vectorizer.transform([text])
    coefs = model.coef_
    classes = model.classes_

    pred_idx = model.predict(vec)[0]
    class_idx = list(classes).index(pred_idx)

    feature_names = vectorizer.get_feature_names_out()
    # Get the feature vector for this text (sparse)
    input_vector = vec.toarray()[0]

    # Multiply feature vector by coef weights for predicted class
    feature_weights = input_vector * coefs[class_idx]

    # Get top weighted features (indices)
    top_indices = feature_weights.argsort()[-top_n:][::-1]

    features = []
    for idx in top_indices:
        if input_vector[idx] > 0:
            features.append(feature_names[idx])
    return features

if st.button("Predict Origin"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0]
        pred_proba = model.predict_proba(input_vec)[0]

        classes = model.classes_
        class_probs = dict(zip(classes, pred_proba))

        # Sort classes by confidence
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)

        main_pred, main_conf = sorted_probs[0]
        second_pred, second_conf = sorted_probs[1]

        # Explanation of features
        features = explain_features(user_input, vectorizer, model)

        st.success(f"Predicted Origin: {main_pred}")

        # Confidence threshold to show secondary prediction (e.g., within 10% of main confidence)
        threshold = 0.10

        st.markdown(f"**Prediction confidence:** {main_conf*100:.2f}%")

        if features:
            feat_list = ", ".join(f"'{f}'" for f in features)
            st.markdown(f"**Why this prediction?**\nThis prediction is based on writing features such as: {feat_list}.")
        else:
            st.markdown("**Why this prediction?**\nThe model identified patterns typical for this origin.")

        # Show second choice only if close enough confidence
        if (main_conf - second_conf) <= threshold:
            st.markdown(f"**Possible secondary origin:** {second_pred} ({second_conf*100:.2f}%) — due to overlapping writing features.")

    else:
        st.error("Please enter some text to analyze.")
