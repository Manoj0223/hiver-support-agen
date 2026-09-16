import argparse

from src.classifier import train_classifier, predict_intents
from src.config import TOP_K
from src.data_loader import load_apple_support_pairs
from src.escalation import should_escalate
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
    # 1. Train intent classifier
    # ---------------------------------------------------------

    print("\nTraining intent classifier...")

    classifier = train_classifier(
        pairs["clean_customer_text"],
        pairs["intent"] if "intent" in pairs.columns else _create_fallback_labels(
            pairs["clean_customer_text"]
        ),
    )

    prediction, confidence = predict_intents(
        classifier,
        [args.query],
    )

    predicted_intent = prediction[0]
    predicted_confidence = float(confidence[0])

    # ---------------------------------------------------------
    # 2. Build historical retrieval index
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
    # 3. Generate response
    # ---------------------------------------------------------

    reply = generate_reply_from_results(
        args.query,
        predicted_intent,
        results,
    )

    # ---------------------------------------------------------
    # 4. Escalation decision
    # ---------------------------------------------------------

    escalate, escalation_reason = should_escalate(
        args.query,
        best_similarity,
    )

    # ---------------------------------------------------------
    # Output
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


def _create_fallback_labels(texts):
    """
    Temporary fallback for the demo.

    The production evaluation uses the weak-label taxonomy created
    during the project. This fallback keeps run.py executable before
    the labeling module is added.
    """

    labels = []

    for text in texts:
        text = text.lower()

        if any(
            word in text
            for word in [
                "battery",
                "charging",
                "charge",
            ]
        ):
            labels.append("battery_charging")

        elif any(
            word in text
            for word in [
                "update",
                "ios",
                "software",
            ]
        ):
            labels.append("ios_update")

        elif any(
            word in text
            for word in [
                "wifi",
                "bluetooth",
                "internet",
                "network",
            ]
        ):
            labels.append("wifi_connectivity")

        elif any(
            word in text
            for word in [
                "app",
                "application",
                "crash",
                "freezing",
            ]
        ):
            labels.append("apps")

        elif any(
            word in text
            for word in [
                "icloud",
                "itunes",
                "apple music",
                "app store",
            ]
        ):
            labels.append("apple_services")

        elif any(
            word in text
            for word in [
                "payment",
                "billing",
                "refund",
                "subscription",
                "apple id",
            ]
        ):
            labels.append("account_payment")

        elif any(
            word in text
            for word in [
                "screen",
                "keyboard",
                "button",
                "overheating",
            ]
        ):
            labels.append("device_hardware")

        elif any(
            word in text
            for word in [
                "how do i",
                "how can i",
                "how to",
            ]
        ):
            labels.append("how_to")

        else:
            labels.append("other")

    return labels


if __name__ == "__main__":
    main()
