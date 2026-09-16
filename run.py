import argparse

from src.classifier import train_classifier, predict_intents
from src.config import DEMO_SAMPLE_SIZE, TOP_K
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

    # ---------------------------------------------------------
    # 1. Load Apple Support conversations
    # ---------------------------------------------------------

    print("Loading Apple Support conversations...")

    pairs = load_apple_support_pairs(args.data)

    print(f"Usable Apple Support pairs: {len(pairs):,}")

    # ---------------------------------------------------------
    # 2. Sample data for interactive demo
    # ---------------------------------------------------------

    if len(pairs) > DEMO_SAMPLE_SIZE:
        pairs = pairs.sample(
            n=DEMO_SAMPLE_SIZE,
            random_state=42,
        ).reset_index(drop=True)

    print(
        f"Demo training/retrieval rows: {len(pairs):,}"
    )

    # ---------------------------------------------------------
    # 3. Create weak intent labels
    # ---------------------------------------------------------

    print("\nCreating weak intent labels...")

    pairs["intent"] = create_weak_labels(
        pairs["clean_customer_text"]
    )

    # ---------------------------------------------------------
    # 4. Train intent classifier
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
    # 5. Build historical retrieval index
    # ---------------------------------------------------------

    print("Building historical retrieval index...")

    retriever = HistoricalRetriever(
        pairs["clean_customer_text"],
        pairs["support_response"],
    )

    results = retriever.search(
        args.query,
        top_k=TOP_K,
    )

    best_similarity = results[0]["similarity"]

    # ---------------------------------------------------------
    # 6. Generate draft response
    # ---------------------------------------------------------

    reply = generate_reply_from_results(
        args.query,
        predicted_intent,
        results,
    )

    # ---------------------------------------------------------
    # 7. Decide escalation
    # ---------------------------------------------------------

    escalate, escalation_reason = should_escalate(
        args.query,
        best_similarity,
    )

    # ---------------------------------------------------------
    # 8. Display agent result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 60)

    print("\nCustomer message:")
    print(args.query)

    print("\nPredicted intent:")
    print(predicted_intent)

    print("\nIntent confidence:")
    print(f"{predicted_confidence:.3f}")

    print("\nBest historical similarity:")
    print(f"{best_similarity:.3f}")

    print("\nDraft reply:")

    if reply:
        print(reply)
    else:
        print("[No automated reply generated]")

    print("\nEscalation:")
    print("YES" if escalate else "NO")

    print("\nEscalation reason:")
    print(escalation_reason)

    # ---------------------------------------------------------
    # 9. Display retrieved historical examples
    # ---------------------------------------------------------

    print("\nTop historical matches:")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. similarity={result['similarity']:.3f} | "
            f"{result['customer_text'][:120]}"
        )

        if result.get("support_response"):
            print(
                f"   Historical response: "
                f"{result['support_response'][:200]}"
            )


if __name__ == "__main__":
    main()
