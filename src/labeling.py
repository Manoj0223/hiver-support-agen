INTENTS = {
    "ios_update": "Problems with iOS or software updates, including failed updates, bugs after updating, or update-related issues.",
    "battery_charging": "Battery draining quickly, battery health issues, charging problems, or device not charging.",
    "wifi_connectivity": "Wi-Fi, Bluetooth, cellular, internet, or other connectivity problems.",
    "apps": "Problems with apps, including apps crashing, freezing, not opening, or behaving incorrectly.",
    "device_hardware": "Physical device problems such as screen, keyboard, buttons, overheating, or other hardware issues.",
    "apple_services": "Problems with Apple services such as iCloud, Apple Music, iTunes, App Store, Photos, or related services.",
    "account_payment": "Apple ID, account access, billing, subscriptions, purchases, or payment-related issues.",
    "how_to": "Questions asking how to perform a feature, setting, or task on an Apple device.",
    "other": "Messages that do not clearly belong to another intent or do not contain enough information to classify."
}


def assign_initial_intent(text):
    t = str(text).lower()

    if any(k in t for k in [
        "battery", "charge", "charging", "charger", "power"
    ]):
        return "battery_charging"

    if any(k in t for k in [
        "ios", "update", "updated", "upgrade"
    ]):
        return "ios_update"

    if any(k in t for k in [
        "wifi", "wi-fi", "bluetooth", "cellular",
        "mobile data", "internet", "network", "signal"
    ]):
        return "wifi_connectivity"

    if any(k in t for k in [
        "apple id", "password", "sign in", "login",
        "account", "payment", "billing", "charged",
        "refund", "subscription", "purchase"
    ]):
        return "account_payment"

    if any(k in t for k in [
        "icloud", "itunes", "apple music", "app store",
        "imessage", "facetime", "apple pay", "photos"
    ]):
        return "apple_services"

    if any(k in t for k in [
        "app", "application", "snapchat", "instagram",
        "facebook", "twitter", "spotify", "crash", "freeze"
    ]):
        return "apps"

    if any(k in t for k in [
        "screen", "display", "keyboard", "button",
        "camera", "speaker", "microphone", "overheat",
        "broken", "crack", "not turning on"
    ]):
        return "device_hardware"

    if any(k in t for k in [
        "how do i", "how can i", "how to", "where do i",
        "can i", "setting"
    ]):
        return "how_to"

    return "other"


def create_weak_labels(texts):
    return texts.apply(assign_initial_intent)
