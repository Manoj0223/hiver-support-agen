import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader import load_apple_support_pairs
from src.labeling import create_weak_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--golden", required=True)
    args = parser.parse_args()

    print("Loading Apple Support conversations...")

    pairs = load_apple_support_pairs(args.data)

    print(f"Apple Support pairs: {len(pairs)}")

    # Create weak labels exactly as used in the development pipeline
    pairs["intent"] = create_weak_labels(
        pairs["clean_customer_text"]
    )

    # Load golden set
    golden = pd.read_csv(args.golden)

    print(f"Golden rows: {len(golden)}")

    # Prevent golden-set leakage
    if "customer_tweet_id" in golden.columns:
        golden_ids = set(
            golden["customer_tweet_id"].astype(str)
        )

        pairs = pairs[
            ~pairs["customer_tweet_id"]
            .astype(str)
            .isin(golden_ids)
        ].copy()

    print(f"Leakage-safe training pool: {len(pairs)}")

    # --------------------------------------------------
    # Kaggle-style train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        pairs["clean_customer_text"],
        pairs["intent"],
        test_size=0.2,
        random_state=42,
        stratify=pairs["intent"]
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Internal test rows: {len(X_test)}")

    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    classifier.fit(X_train_tfidf, y_train)

    # --------------------------------------------------
    # Golden-set evaluation
    # --------------------------------------------------

    golden_text = golden["customer_text"].fillna("").astype(str)

    X_golden_tfidf = tfidf.transform(golden_text)

    predictions = classifier.predict(X_golden_tfidf)
    probabilities = classifier.predict_proba(X_golden_tfidf)

    confidence = probabilities.max(axis=1)

    y_true = golden["gold_intent"].astype(str)

    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted"
    )

    print()
    print("=" * 50)
    print("GOLDEN SET RESULTS")
    print("=" * 50)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print()
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Save detailed results
    # --------------------------------------------------

    results = golden.copy()

    results["pred_intent"] = predictions
    results["pred_confidence"] = confidence

    results.to_csv(
        "evaluation_results.csv",
        index=False
    )

    print("Saved: evaluation_results.csv")


if __name__ == "__main__":
    main()
