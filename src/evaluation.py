from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


def evaluate_classifier(y_true, y_pred):
    """
    Calculate intent classification metrics.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def get_classification_report(y_true, y_pred):
    """
    Return a detailed per-intent classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )


def majority_baseline(y_train, y_test):
    """
    Trivial baseline: always predict the majority training class.
    """

    majority_class = y_train.value_counts().idxmax()

    predictions = [majority_class] * len(y_test)

    return {
        "predictions": predictions,
        "majority_class": majority_class,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }
