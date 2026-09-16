import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.classifier import train_classifier, predict_intents
from src.data_loader import load_apple_support_pairs
from src.labeling import create_weak_labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Apple Support intent classifier"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to twcs.csv",
    )

    parser.add_argument(
        "--golden",
        required=True,
        help="Path to golden evaluation CSV",
    )

    args = parser.parse_args()

    print("Loading dataset...")
    pairs = load_apple_support_pairs(args.data)

    print(f"Apple Support pairs: {len(pairs):,}")

    # Create weak labels for training.
    pairs["intent"] = create_weak_labels(
        pairs["clean_customer_text"]
    )

    golden = pd.read_csv(args.golden)

    # Exclude golden examples from training when IDs are available.
    if "customer_tweet_id" in golden.columns:
        golden_ids = set(
            golden["customer_tweet_id"].astype(str)
        )

        pairs = pairs[
            ~pairs["customer_tweet_id"]
            .astype(str)
            .isin(golden_ids)
        ].copy()

    print(f"Training rows: {len(pairs):,}")
    print(f"Golden rows: {len(golden):,}")

    # Train classifier.
    model = train_classifier(
        pairs["clean_customer_text"],
        pairs["intent"],
    )

    # Prepare golden messages.
    golden_text = golden["customer_text"].fillna("").astype(str)

    predictions, confidence = predict_intents(
        model,
        golden_text,
    )

    y_true = golden["gold_intent"].astype(str)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 50)
    print("GOLDEN SET RESULTS")
    print("=" * 50)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # Save predictions.
    output = golden.copy()

    output["pred_intent"] = predictions
    output["pred_confidence"] = confidence

    output.to_csv(
        "evaluation_results.csv",
        index=False,
    )

    print("\nSaved: evaluation_results.csv")


if __name__ == "__main__":
    main()
