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
    Exact weak-labeling logic used in the Kaggle development pipeline.
    """

    text = str(text).lower()

    if any(k in text for k in [
        "update", "ios", "software update", "upgrade", "install ios"
    ]):
        return "ios_update"

    if any(k in text for k in [
        "battery", "charging", "charger", "charge", "battery health"
    ]):
        return "battery_charging"

    if any(k in text for k in [
        "wifi", "wi-fi", "bluetooth", "internet", "cellular",
        "mobile data", "network", "signal"
    ]):
        return "wifi_connectivity"

    if any(k in text for k in [
        "app", "application", "crash", "freezing", "freeze",
        "not opening", "won't open"
    ]):
        return "apps"

    if any(k in text for k in [
        "screen", "display", "keyboard", "button", "overheating",
        "overheat", "broken", "cracked", "speaker", "camera"
    ]):
        return "device_hardware"

    if any(k in text for k in [
        "icloud", "itunes", "apple music", "app store",
        "photos", "facetime", "imessage", "apple pay"
    ]):
        return "apple_services"

    if any(k in text for k in [
        "apple id", "password", "login", "sign in", "account",
        "billing", "payment", "subscription", "refund", "charged"
    ]):
        return "account_payment"

    if any(k in text for k in [
        "how do i", "how can i", "how to", "where can i",
        "can i change", "how can"
    ]):
        return "how_to"

    return "other"


def create_weak_labels(texts):
    """Create weak labels using the Kaggle development rules."""

    return [
        assign_initial_intent(text)
        for text in texts
    ]
