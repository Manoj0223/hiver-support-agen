from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_classifier():
    """
    Build the TF-IDF + Logistic Regression intent classifier.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_classifier(texts, labels):
    """
    Train the intent classifier.

    Parameters
    ----------
    texts : iterable
        Cleaned customer messages.
    labels : iterable
        Intent labels.

    Returns
    -------
    Pipeline
        Trained classifier.
    """
    model = build_classifier()
    model.fit(texts, labels)
    return model


def predict_intents(model, texts):
    """
    Predict intents and confidence scores.

    Returns
    -------
    predictions : array
        Predicted intent for each message.
    confidence : array
        Maximum class probability for each prediction.
    """
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    confidence = probabilities.max(axis=1)

    return predictions, confidence
