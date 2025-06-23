import streamlit as st
import joblib
import numpy as np

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

# Map n-grams to user-friendly explanations (expandable)
EXPLANATION_MAP = {
    "n't": "use of negative contractions like 'can't' or 'won't'",
    "'re": "use of contractions like 'you're' or 'they're'",
    "'ll": "use of future tense contractions like 'I'll' or 'you'll'",
    "ain": "informal negations like 'ain't'",
    "yo": "informal/slang expressions such as 'yo'",
    "ya": "casual speech like 'ya' instead of 'you'",
    " u ": "direct address pronouns like 'you'",
    "ing": "frequent use of '-ing' verb forms",
    "the": "use of the definite article 'the'",
    " and": "use of conjunction 'and'",
    " of ": "common preposition 'of'",
    " in ": "common preposition 'in'",
    " is ": "use of the verb 'is'",
    " to ": "infinitive marker 'to'",
    " th": "common letter pattern as in 'that', 'this', 'the'",
    " fo": "common word starts like 'for' or 'from'",
}

# Generic style hints per region if no strong features found
GENERIC_STYLES = {
    "UK": "often uses formal British English spelling and phrasing.",
    "US": "frequently uses American English spelling and colloquial expressions.",
    "Australia": "may include Australian slang and idioms.",
    "Spain": "shows influences of Spanish syntax or English as a second language.",
    "Japan": "reflects indirect phrasing or English influenced by Japanese grammar.",
    "Canada": "mixes British and American English influences.",
}

def explain_prediction(text):
    vec = vectorizer.transform([text])
    coef = model.coef_
    features = vectorizer.get_feature_names_out()
    pred_class = model.predict(vec)[0]
    class_idx = list(model.classes_).index(pred_class)
    feature_contributions = coef[class_idx] * vec.toarray()[0]

    # Sort by absolute contribution descending
    sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]

    explanations = []
    used_explanations = set()
    count = 0

    significance_threshold = 0.01  # Raise threshold to filter noise

    for i in sorted_indices:
        weight = feature_contributions[i]
        if abs(weight) < significance_threshold:
            continue

        feat = features[i].lower()

        # Skip meaningless very short fragments or single letters
        if len(feat.strip()) < 2:
            continue
        if feat in ['n t', ' t ', 'nt']:
            continue

        # Map to friendly explanation if possible
        explanation = None
        for key in EXPLANATION_MAP:
            if key in feat:
                explanation = EXPLANATION_MAP[key]
                break

        if explanation is None:
            # Format patterns more naturally: e.g. 'usage of pattern "xyz"'
            explanation = f"usage of the pattern '{feat}'"

        if explanation not in used_explanations:
            explanations.append(explanation)
            used_explanations.add(explanation)
            count += 1

        if count >= 5:
            break

    if not explanations:
        style_hint = GENERIC_STYLES.get(pred_class, "has distinctive regional writing features.")
        explanations.append(f"general writing style hint: {style_hint}")

    explanation_text = (
        f"The text is predicted to be from {pred_class}. "
        f"This is because it shows features such as: "
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

