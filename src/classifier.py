from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_classifier():
    """
    Build the intent classifier.

    Word + character n-grams help with short/noisy Twitter messages,
    while class balancing prevents frequent intents from dominating.
    """

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

    return predictions, confidence
