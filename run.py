import argparse

from src.classifier import train_classifier, predict_intents
from src.config import TOP_K
from src.data_loader import load_apple_support_pairs
from src.escalation import should_escalate
from src.labeling import create_weak_labels
from src.retrieval import HistoricalRetriever
from src.response_generator import generate_reply_from_results


def main():
    parser = argparse.ArgumentParser(
        description="Apple Support AI Agent"
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Path to twcs.csv",
    )

    parser.add_argument(
        "--query",
        default="My iPhone battery is draining very quickly.",
        help="Customer message to process.",
    )

    args = parser.parse_args()

    print("Loading Apple Support conversations...")

    pairs = load_apple_support_pairs(args.data)

    print(f"Usable Apple Support pairs: {len(pairs):,}")

    # ---------------------------------------------------------
    # 1. Create weak labels for classifier training
    # ---------------------------------------------------------

    print("\nCreating weak intent labels...")

    pairs["intent"] = create_weak_labels(
        pairs["clean_customer_text"]
    )

    # ---------------------------------------------------------
    # 2. Train intent classifier
    # ---------------------------------------------------------

    print("Training intent classifier...")

    classifier = train_classifier(
        pairs["clean_customer_text"],
        pairs["intent"],
    )

    prediction, confidence = predict_intents(
        classifier,
        [args.query],
    )

    predicted_intent = prediction[0]
    predicted_confidence = float(confidence[0])

    # ---------------------------------------------------------
    # 3. Build historical retrieval index
    # ---------------------------------------------------------

    print("Building historical retrieval index...")

    retriever = HistoricalRetriever(
        pairs["clean_customer_text"]
    )

    results = retriever.search(
        args.query,
        top_k=TOP_K,
    )

    best_similarity = results[0]["similarity"]

    # ---------------------------------------------------------
    # 4. Generate reply
    # ---------------------------------------------------------

    reply = generate_reply_from_results(
        args.query,
        predicted_intent,
        results,
    )

    # ---------------------------------------------------------
    # 5. Escalation decision
    # ---------------------------------------------------------

    escalate, escalation_reason = should_escalate(
        args.query,
        best_similarity,
    )

    # ---------------------------------------------------------
    # 6. Display result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 60)

    print(f"\nCustomer message:")
    print(args.query)

    print(f"\nPredicted intent:")
    print(predicted_intent)

    print(f"Intent confidence:")
    print(f"{predicted_confidence:.3f}")

    print(f"\nBest historical similarity:")
    print(f"{best_similarity:.3f}")

    print("\nDraft reply:")
    print(reply if reply else "[No automated reply generated]")

    print("\nEscalation:")
    print("YES" if escalate else "NO")

    print(f"Reason:")
    print(escalation_reason)

    print("\nTop historical matches:")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. similarity={result['similarity']:.3f} | "
            f"{result['customer_text'][:120]}"
        )


if __name__ == "__main__":
    main()
