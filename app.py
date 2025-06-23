import streamlit as st
import joblib
import numpy as np

# Load new model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title("Writing Origin Predictor")

user_input = st.text_area("Enter a sample of your writing:")

EXPLANATION_MAP = {
    "n't": "use of negative contractions like 'can't' or 'won't'",
    "'re": "use of contractions like 'you're' or 'they're'",
    "'ll": "use of future tense contractions like 'I'll' or 'you'll'",
    "ain": "informal negations like 'ain't'",
    "yo": "informal/slang expressions such as 'yo'",
    "ya": "casual speech like 'ya' instead of 'you'",
    " u ": "direct address pronouns like 'you'",
    "ing": "frequent use of '-ing' verb forms",
    " th": "common letter pattern as in 'that', 'this', 'the'",
    " fo": "common word starts like 'for' or 'from'",
    "bhai": "usage of informal Indian terms like 'bhai'",
    "ji": "polite suffix commonly used in Indian English like 'ji'",
    "sir": "formal address often seen in Indian English",
    "lah": "colloquial particle often used in Singaporean/Malaysian English",
    # China & Russia features examples:
    "no": "typical negation style seen in Russian English",
    "very": "common intensifier usage in Chinese English",
    "go": "frequent verb form usage seen in Chinese English",
    "ok": "common filler or agreement word in Chinese English",
    "is": "common to both but with distinct patterns",
}

COMMON_TOKENS_TO_IGNORE = {
    "the", "and", "is", "of", "in", "to", "a", "that", "it", "for"
}

GENERIC_STYLES = {
    "UK": "often uses formal British English spelling and phrasing.",
    "US": "frequently uses American English spelling and colloquial expressions.",
    "Australia": "may include Australian slang and idioms.",
    "Spain": "shows influences of Spanish syntax or English as a second language.",
    "Japan": "reflects indirect phrasing or English influenced by Japanese grammar.",
    "Canada": "mixes British and American English influences.",
    "India": "often includes formal phrasing, Hindi/Urdu loanwords, and respectful address.",
    "Singapore": "includes colloquial particles like 'lah' and mix of English dialects.",
    "China": "shows influences of Chinese English, often marked by unique syntax and phrasing.",
    "Russia": "reflects Russian English traits, including direct phrasing and unique negation forms.",
}

def explain_prediction(text):
    vec = vectorizer.transform([text])
    coef = model.coef_
    features = vectorizer.get_feature_names_out()
    pred_class = model.predict(vec)[0]
    class_idx = list(model.classes_).index(pred_class)
    feature_contributions = coef[class_idx] * vec.toarray()[0]

    sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]

    explanations = []
    used_explanations = set()
    count = 0

    significance_threshold = 0.005

    for i in sorted_indices:
        weight = feature_contributions[i]
        if abs(weight) < significance_threshold:
            continue

        feat = features[i].lower().strip()
        if feat in COMMON_TOKENS_TO_IGNORE:
            continue
        if len(feat) < 2:
            continue
        if feat in ['n t', ' t ', 'nt']:
            continue

        explanation = None
        for key in EXPLANATION_MAP:
            if key in feat:
                explanation = EXPLANATION_MAP[key]
                break

        if explanation is None:
            explanation = f"usage of the pattern '{feat}'"

        if explanation not in used_explanations:
            explanations.append(explanation)
            used_explanations.add(explanation)
            count += 1

        if count >= 7:
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
        vec = vectorizer.transform([user_input])
        preds = model.predict(vec)
        probs = model.predict_proba(vec)[0]
        prediction = preds[0]

        st.success(f"Predicted Origin: {prediction}")
        st.markdown("### Prediction confidence scores:")
        for cls, prob in zip(model.classes_, probs):
            st.write(f"{cls}: {prob:.2%}")

        explanation = explain_prediction(user_input)
        st.markdown("### Why this prediction?")
        st.write(explanation)
    else:
        st.error("Please enter some text to analyze.")



