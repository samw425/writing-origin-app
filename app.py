import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

def explain_prediction(text, model, vectorizer, top_n=5):
    vec = vectorizer.transform([text])
    coefs = model.coef_
    classes = model.classes_
    prediction = model.predict(vec)[0]
    class_idx = list(classes).index(prediction)

    feature_names = vectorizer.get_feature_names_out()
    text_vec = vec.toarray()[0]
    contributions = coefs[class_idx] * text_vec

    top_indices = np.argsort(contributions)[-top_n:][::-1]
    top_features = [(feature_names[i], contributions[i]) for i in top_indices if contributions[i] > 0]

    # Explanation map for meaningful n-grams
    explanation_map = {
        "we": "use of inclusive pronouns like 'we'",
        "n't": "frequent use of contractions such as 'don't' and 'can't'",
        "can't": "common negation contractions like 'can't'",
        "won't": "common negation contractions like 'won't'",
        "the": "usage of the definite article 'the'",
        "you": "direct address to the reader using 'you'",
        "g'day": "informal greeting common in Australian English",
        "mate": "friendly term often used in British and Australian English",
        "y'all": "colloquial plural 'you' common in Southern US English",
        "barbecue": "cultural reference typical in Southern US English",
        " in ": "typical use of the preposition 'in'",
        " is ": "usage of the verb 'is'",
        # add more as needed
    }

    seen = set()
    explanations = []

    for feat, val in top_features:
        clean_feat = feat.strip().replace("'", "").replace(" ", "").lower()

        # Skip short or meaningless ngrams
        if len(clean_feat) < 3:
            continue

        # Attempt to match explanation map by checking for keys contained in the feature
        matched = False
        for key in explanation_map:
            if key.replace(" ", "") in clean_feat:
                if key not in seen:
                    explanations.append(explanation_map[key])
                    seen.add(key)
                matched = True
                break
        if not matched and feat.strip() not in seen:
            # Optionally skip unknown patterns or include generic description
            # explanations.append(f"usage of pattern '{feat.strip()}'")
            pass

    if explanations:
        explanation_text = (
            f"The text shows features common to writing from {prediction}, including "
            + ", ".join(explanations[:-1])
            + (", and " if len(explanations) > 1 else "")
            + explanations[-1]
            + "."
        )
    else:
        explanation_text = f"The text shows features common to writing from {prediction}."

    return explanation_text

st.title("Writing Origin Predictor with Plain English Explanation")

user_input = st.text_area("Enter a sample of your writing:")

if st.button("Analyze"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Origin: {prediction}")

        explanation = explain_prediction(user_input, model, vectorizer)
        st.text_area("Why this prediction?", explanation, height=150)
    else:
        st.error("Please enter some text to analyze.")
