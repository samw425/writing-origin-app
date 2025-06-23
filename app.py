import streamlit as st
import joblib
import numpy as np

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

# Mapping common n-grams to friendly explanations (expand as needed)
EXPLANATION_MAP = {
    "n't": "use of negative contractions like 'n't' (e.g., can't, won't)",
    "'re": "use of contractions like 'you're' or 'they're'",
    "'ll": "use of future tense contractions like 'I'll' or 'you'll'",
    "ain": "use of informal negations like 'ain't'",
    "yo": "presence of informal/slang terms such as 'yo'",
    "ya": "informal speech pattern like 'ya' instead of 'you'",
    " u ": "use of direct address pronouns like 'you'",
    "ing": "frequent use of present participles ending with 'ing'",
    "the": "usage of the definite article 'the'",
    " and": "use of conjunction 'and' to link phrases",
    " of ": "use of preposition 'of' common in formal writing",
    " in ": "preposition 'in' usage",
    " is ": "usage of the verb 'is'",
    " to ": "infinitive marker 'to'",
    " th": "common letter pattern 'th', as in 'that', 'this', 'the'",
    " fo": "common start of words like 'for', 'from'",
}

def explain_prediction(text):
    vec = vectorizer.transform([text])
    coef = model.coef_
    features = vectorizer.get_feature_names_out()
    pred_class = model.predict(vec)[0]
    class_idx = list(model.classes_).index(pred_class)
    feature_contributions = coef[class_idx] * vec.toarray()[0]

    # Sort indices by absolute contribution descending
    sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]

    explanations = []
    used_explanations = set()
    count = 0

    # Threshold for significance of features
    significance_threshold = 0.005

    for i in sorted_indices:
        weight = feature_contributions[i]
        if abs(weight) < significance_threshold:
            continue

        feat = features[i]
        # Clean ngram string for matching
        feat_clean = feat.lower()

        # Check if this ngram maps to a friendly explanation
        explanation = None
        for key in EXPLANATION_MAP:
            if key in feat_clean:
                explanation = EXPLANATION_MAP[key]
                break

        if explanation is None:
            # Generic fallback if no mapping found
            explanation = f"usage of the pattern '{feat}'"

        # Avoid duplicates
        if explanation not in used_explanations:
            explanations.append(explanation)
            used_explanations.add(explanation)
            count += 1

        if count >= 7:
            break

    # If no distinctive features found, fallback to generic style hint per origin
    if not explanations:
        generic_styles = {
            "UK": "tends to use more formal British English spelling and phrasing.",
            "US": "often uses American English spelling and colloquial expressions.",
            "Australia": "may include Australian slang and idioms.",
            "Spain": "writing shows influences of Spanish syntax or English learned as a second language.",
            "Japan": "may reflect indirect phrasing or English influenced by Japanese grammar.",
            "Canada": "includes a mix of British and American English influences.",
        }
        style_hint = generic_styles.get(pred_class, "has unique linguistic features characteristic of its region.")
        explanations.append(f"general writing style hint: {style_hint}")

    explanation_text = (
        f"The text is predicted to be from {pred_class}. "
        f"Why this prediction? Because the text shows features including: "
        + ", ".join(explanations) + "."
    )

    return explanation_text

if st.button("Predict Origin"):
    if user_input.strip():
        prediction = model.predict(vectorizer.transform([user_input]))[0]
        explanation = explain_prediction(user_input)
        st.success(f"Predicted Origin: {prediction}")
        st.markdown("### Why this prediction?")
        st.write(explanation)
    else:
        st.error("Please enter some text to analyze.")
