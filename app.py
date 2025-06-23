import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Helper: Translate n-grams into plain English explanations
def explain_ngrams(ngrams):
    explanations = []
    for gram in ngrams:
        if "is a" in gram:
            explanations.append("frequent use of formal phrases like 'this is a...'")
        elif "s a" in gram:
            explanations.append("use of linking structures such as 'it's a...'")
        elif "a t" in gram:
            explanations.append("structured phrasing like 'at the...'")
        elif "s i" in gram:
            explanations.append("phrasing like 'is in...' or similar")
        elif "the" in gram:
            explanations.append("frequent use of the article 'the'")
        elif "ing" in gram:
            explanations.append("use of continuous tense verbs ending in -ing")
        elif "n't" in gram or " not" in gram:
            explanations.append("contractions or negation patterns like 'don't', 'not going'")
        elif "you" in gram:
            explanations.append("direct address or conversational tone using 'you'")
        elif "mate" in gram:
            explanations.append("informal British or Australian tone using words like 'mate'")
        else:
            explanations.append(f"common stylistic pattern '{gram.strip()}'")
    return explanations[:3]

st.title("🌍 Writing Origin Analyzer")

user_input = st.text_area("Enter a sample of your writing:", height=200)

if st.button("Predict Origin"):
    if user_input.strip():
        # Vectorize and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probs = model.predict_proba(input_vec)[0]
        class_labels = model.classes_
        sorted_indices = np.argsort(probs)[::-1]
        top_indices = sorted_indices[:3]
        
        # Show main prediction
        st.subheader(f"Predicted Origin: {prediction}")
        
        # Top 3 Confidence Scores
        st.markdown("**Top 3 Prediction Confidence Scores:**")
        for idx in top_indices:
            st.write(f"{class_labels[idx]}: {probs[idx]*100:.2f}%")
        
        # Explanation
        st.markdown("### Why this prediction?")
        top_ngrams = input_vec @ vectorizer.transform(vectorizer.get_feature_names_out()).T
        top_ngrams = vectorizer.get_feature_names_out()[np.argsort(top_ngrams.toarray()[0])[::-1][:5]]
        explanation_lines = explain_ngrams(top_ngrams)
        
        if explanation_lines:
            st.write("The model is confident in this prediction based on:")
            for line in explanation_lines:
                st.write(f"- {line}")
        else:
            st.write("The model found patterns typical of this region, but could not determine clear stylistic features.")
    else:
        st.error("Please enter some text to analyze.")
