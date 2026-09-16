from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_classifier():
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    classifier = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=2.0
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])


def train_classifier(texts, labels):
    model = build_classifier()
    model.fit(texts, labels)
    return model


def predict_intents(model, texts):

    predictions = model.predict(texts)

    probabilities = model.predict_proba(texts)
    confidence = probabilities.max(axis=1)

    corrected_predictions = []

    for text, prediction in zip(texts, predictions):

        text_lower = str(text).lower()

        # Conservative app-specific signals
        clear_app_signals = [
            "app crashes",
            "apps crash",
            "app crash",
            "app crashing",
            "apps crashing",
            "app freeze",
            "app freezes",
            "apps freeze",
            "apps freezing",
            "app is frozen",
            "app won't open",
            "app wont open",
            "app not opening",
            "app doesn't open",
            "app doesnt open",
            "application crashes",
            "application crash",
            "application not opening",
            "application won't open",
            "application wont open"
        ]

        # If the model predicts apps but there is no
        # clear app-specific failure, use other.
        if prediction == "apps":
            if not any(
                signal in text_lower
                for signal in clear_app_signals
            ):
                prediction = "other"

        corrected_predictions.append(prediction)

    return (
        corrected_predictions,
        probabilities.max(axis=1)
    )
