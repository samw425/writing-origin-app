import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

if st.button("Predict Origin"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0]
        pred_proba = model.predict_proba(input_vec)[0]

        # Sort indices by probability descending
        sorted_idx = np.argsort(pred_proba)[::-1]
        top_idx = sorted_idx[0]
        second_idx = sorted_idx[1]

        top_origin = model.classes_[top_idx]
        second_origin = model.classes_[second_idx]
        top_conf = pred_proba[top_idx]
        second_conf = pred_proba[second_idx]

        # Get top weighted features for explanation
        coefs = model.coef_[top_idx]
        feature_names = vectorizer.get_feature_names_out()
        input_vec_array = input_vec.toarray()[0]
        
        # Find n-grams present in input with top weights
        present_features = [(feature_names[i], coefs[i]) for i in range(len(coefs)) if input_vec_array[i] > 0]
        # Sort by weight descending
        present_features.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = [f"'{feat[0]}'" for feat in present_features[:5]]

        explanation = f"This prediction is based on writing features such as: {', '.join(top_features)}."

        st.success(f"Predicted Origin: {top_origin}")

        st.markdown("### Why this prediction?")
        st.write(explanation)

        # Show possible secondary if close confidence (within 10% absolute difference)
        if (top_conf - second_conf) < 0.1:
            st.markdown(f"**Possible secondary origin:** {second_origin} — due to overlapping writing features.")
    else:
        st.error("Please enter some text to analyze.")
