import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

def explain_prediction(text, model, vectorizer, top_n=7, min_contribution=0.01):
    vec = vectorizer.transform([text])
    coefs = model.coef_
    classes = model.classes_
    prediction = model.predict(vec)[0]
    class_idx = list(classes).index(prediction)

    feature_names = vectorizer.get_feature_names_out()
    text_vec = vec.toarray()[0]
    contributions = coefs[class_idx] * text_vec

    # Filter features: contribution above threshold & length 3 to 5 (typical ngram range)
    filtered = [
        (feature_names[i], contributions[i]) 
        for i in range(len(contributions))
        if contributions[i] > min_contribution and 3 <= len(feature_names[i].strip()) <= 5
    ]

    # Sort descending by contribution
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]

    # Expanded explanation map with more region- and syntax-specific patterns
    explanation_map = {
        "we": "use of inclusive pronouns like 'we'",
        "n't": "contractions like 'don't' and 'can't'",
        "ain't": "informal negations like 'ain't'",
        "g'day": "Australian informal greeting 'g'day'",
        "mate": "friendly term used in UK/Australia 'mate'",
        "y'all": "Southern US colloquial plural 'y'all'",
        "barbecue": "Southern US cultural reference 'barbecue'",
        "colour": "British English spelling of 'colour'",
        "sushi": "cultural reference common in Japan",
        "football": "sports term used in UK and other English-speaking countries",
        "cheers": "informal British/Australian expression for thanks or goodbye",
        "loo": "British slang for bathroom",
        "bloody": "British intensifier used for emphasis",
        "gotta": "informal contraction 'got to' common in American English",
        "gonna": "informal contraction 'going to'",
        "matey": "friendly British slang similar to 'mate'",
        "youse": "plural you, common in some dialects",
        # Add more as you find them
    }

    explanations = []
    seen = set()
    for feat, val in filtered_sorted:
        clean_feat = feat.strip().replace("'", "").replace(" ", "").lower()
        if clean_feat in seen:
            continue
        seen.add(clean_feat)
        matched = False
        for key in explanation_map:
            if key.replace(" ", "") in clean_feat:
                explanations.append(explanation_map[key])
                matched = True
                break
        # skip unknown patterns to keep explanation clean

    # Build explanation text
    if explanations:
        explanation_text = (
            f"The writing strongly reflects features common to {prediction}, including "
            + ", ".join(explanations[:-1])
            + (", and " if len(explanations) > 1 else "")
            + explanations[-1]
            + "."
        )
    else:
        explanation_text = f"The writing reflects features common to {prediction}."

    confidence = contributions.sum()
    explanation_text += (
        f"\n\nConfidence score: {confidence:.3f} (higher means stronger match)."
    )

    # Add a high-level summary about writing style tendencies
    if confidence > 0.1:
        explanation_text += (
            "\n\nThis suggests a distinctive and consistent writing style "
            "matching that origin."
        )
    elif confidence > 0.05:
        explanation_text += (
            "\n\nThe writing shows some markers of this origin, but also mixed elements."
        )
    else:
        explanation_text += (
            "\n\nThe writing shows only weak markers of this origin and may be inconclusive."
        )

    return explanation_text

st.title("Writing Origin Predictor — Advanced Syntax & Style Analysis")

user_input = st.text_area("Enter a sample of your writing:")

if st.button("Analyze"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Origin: {prediction}")

        explanation = explain_prediction(user_input, model, vectorizer)
        st.text_area("Why this prediction?", explanation, height=220)
    else:
        st.error("Please enter some text to analyze.")
