import re


INTENTS = {
    "ios_update": (
        "Problems with iOS or software updates, including failed updates, "
        "bugs after updating, or update-related issues."
    ),
    "battery_charging": (
        "Battery draining quickly, battery health issues, charging problems, "
        "or device not charging."
    ),
    "wifi_connectivity": (
        "Wi-Fi, Bluetooth, cellular, internet, or other connectivity problems."
    ),
    "apps": (
        "Problems with apps, including apps crashing, freezing, not opening, "
        "or behaving incorrectly."
    ),
    "device_hardware": (
        "Physical device problems such as screen, keyboard, buttons, "
        "overheating, or other hardware issues."
    ),
    "apple_services": (
        "Problems with Apple services such as iCloud, Apple Music, iTunes, "
        "App Store, Photos, or related services."
    ),
    "account_payment": (
        "Apple ID, account access, billing, subscriptions, purchases, "
        "or payment-related issues."
    ),
    "how_to": (
        "Questions asking how to perform a feature, setting, or task "
        "on an Apple device."
    ),
    "other": (
        "Messages that do not clearly belong to another intent or do not "
        "contain enough information to classify."
    ),
}


def assign_initial_intent(text):
    """
    Assign a weak initial intent using keyword-based rules.

    These labels are used for classifier training and are not treated
    as human-verified gold labels.
    """

    text = str(text).lower().strip()

    if not text:
        return "other"

    patterns = {
        "battery_charging": [
            "battery",
            "charging",
            "charge",
            "battery life",
            "battery health",
        ],
        "ios_update": [
            "ios",
            "update",
            "updating",
            "software update",
            "upgrade",
        ],
        "wifi_connectivity": [
            "wifi",
            "wi-fi",
            "bluetooth",
            "cellular",
            "internet",
            "network",
            "connection",
            "connectivity",
        ],
        "apps": [
            "app",
            "application",
            "crash",
            "crashing",
            "freeze",
            "freezing",
            "not opening",
        ],
        "device_hardware": [
            "screen",
            "display",
            "keyboard",
            "button",
            "speaker",
            "camera",
            "overheating",
            "broken",
        ],
        "apple_services": [
            "icloud",
            "itunes",
            "apple music",
            "app store",
            "apple pay",
            "photos",
        ],
        "account_payment": [
            "apple id",
            "password",
            "billing",
            "payment",
            "charged",
            "charge",
            "refund",
            "subscription",
            "purchase",
            "credit card",
            "account locked",
            "sign in",
            "login",
        ],
        "how_to": [
            "how do i",
            "how can i",
            "how to",
            "is there a way",
            "can i change",
            "where can i",
        ],
    }

    # Check more specific intents first.
    priority = [
        "account_payment",
        "battery_charging",
        "ios_update",
        "wifi_connectivity",
        "apple_services",
        "device_hardware",
        "apps",
        "how_to",
    ]

    for intent in priority:
        if any(pattern in text for pattern in patterns[intent]):
            return intent

    return "other"


def create_weak_labels(texts):
    """Create weak intent labels for a collection of messages."""

    return [assign_initial_intent(text) for text in texts]
