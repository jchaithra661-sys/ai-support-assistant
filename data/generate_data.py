"""
Generates a synthetic but realistic customer-support dataset for the project:
  - data/tickets.csv            : labeled tickets (text, intent, sentiment_label)
  - data/faq_knowledge_base.json: the knowledge base the RAG retriever searches over

Why synthetic data?
The sandbox this was built in has no outbound access to Kaggle/HuggingFace, so the
dataset here is hand-authored to be realistic and offline-reproducible. For a stronger
portfolio piece, swap this file's output for a real public dataset — see the
"Level up" section in README.md for exact instructions (e.g. the Bitext Customer
Support LLM dataset on Hugging Face, ~27k labeled tickets).
"""
import json
import random
import csv
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent
PRODUCTS = ["the mobile app", "your website", "the desktop dashboard", "my subscription plan", "the checkout page"]
ORDER_IDS = [f"#{n}" for n in random.sample(range(10000, 99999), 40)]

# --- Base sentences per intent, each tagged with a sentiment ------------------
# sentiment: negative / neutral / positive (kept independent of intent on purpose,
# so the classifier has to learn from wording, not just topic)

TEMPLATES = {
    "billing": {
        "negative": [
            "I was charged twice for {product} and nobody has refunded me yet, this is unacceptable.",
            "My invoice for order {order} is completely wrong and support has ignored two emails.",
            "You keep billing me after I cancelled my plan, fix this immediately.",
            "The price on {product} changed without any warning and now I'm being overcharged.",
        ],
        "neutral": [
            "Can you explain the charge on my statement for order {order}?",
            "Where can I download the invoice for {product}?",
            "I'd like to update the billing address on my account.",
            "What payment methods do you accept for {product}?",
        ],
        "positive": [
            "Thanks for sorting out my refund for order {order} so quickly!",
            "Just wanted to say the new pricing on {product} is very fair, appreciate it.",
            "Billing support was fast and helpful today, thank you.",
        ],
    },
    "technical_support": {
        "negative": [
            "{product} keeps crashing every time I try to log in, this is really frustrating.",
            "I'm getting an error 500 on {product} and it's been broken for two days.",
            "Nothing loads on {product} anymore, I've tried restarting three times.",
            "The sync feature on {product} lost all my data, I'm furious.",
        ],
        "neutral": [
            "How do I reset my two-factor authentication for {product}?",
            "Is there a known issue with {product} on Safari?",
            "Can you walk me through installing {product} on a second device?",
            "What are the minimum system requirements for {product}?",
        ],
        "positive": [
            "The bug I reported on {product} last week is fixed now, great turnaround!",
            "Support walked me through the {product} issue patiently, really appreciated it.",
            "{product} has been running smoothly since the last update, nice work.",
        ],
    },
    "account_login": {
        "negative": [
            "I can't log into my account and the password reset link never arrives, very annoying.",
            "My account got locked for no reason and I've lost access to order {order}.",
            "I've been trying to verify my email for {product} for a week now with no luck.",
        ],
        "neutral": [
            "How do I change the email linked to my account?",
            "I want to merge two accounts I accidentally created.",
            "Can you tell me how to enable single sign-on for {product}?",
            "What's the process to recover a username I forgot?",
        ],
        "positive": [
            "Account recovery was surprisingly quick, thank you for the help!",
            "Loved how simple it was to set up two-factor authentication.",
        ],
    },
    "cancellation_refund": {
        "negative": [
            "I want to cancel my subscription immediately, {product} is not worth the money.",
            "Refund my order {order} right now, this is the third time it's arrived broken.",
            "Cancelling has been impossible, the button on {product} just doesn't work.",
        ],
        "neutral": [
            "What's your refund policy for {product}?",
            "How long does a refund for order {order} usually take to process?",
            "Can I pause my subscription instead of cancelling it outright?",
            "I'd like to downgrade my plan starting next month.",
        ],
        "positive": [
            "My refund for order {order} landed in my account today, thanks for the quick handling.",
            "Cancelling was painless, appreciate the no-questions-asked policy.",
        ],
    },
    "shipping_delivery": {
        "negative": [
            "Order {order} was supposed to arrive last week and there's still no tracking update.",
            "My package for order {order} arrived damaged and support hasn't responded.",
            "Delivery for {product} accessories has been delayed three times now, this is ridiculous.",
        ],
        "neutral": [
            "Can you give me a tracking update for order {order}?",
            "Do you ship {product} accessories internationally?",
            "What courier do you use for orders over $100?",
            "Can I change the delivery address for order {order} after placing it?",
        ],
        "positive": [
            "Order {order} arrived a day early, really impressed with the shipping speed!",
            "Packaging for my last order was excellent, thank you for taking care.",
        ],
    },
    "product_info": {
        "negative": [
            "The description for {product} doesn't match what I actually received, misleading.",
            "None of the specs listed for {product} online seem accurate, disappointing.",
        ],
        "neutral": [
            "Does {product} support integration with third-party calendars?",
            "What's the difference between the free and paid tiers of {product}?",
            "Is there a student discount available for {product}?",
            "Can you tell me if {product} is available in the EU region?",
        ],
        "positive": [
            "{product} has more features than I expected for the price, really happy with it.",
            "Just wanted to say {product} has made my workflow so much easier, thank you!",
        ],
    },
}

FAQ_KB = [
    {"id": "faq_001", "category": "billing", "question": "How do I get a refund for a duplicate charge?",
     "answer": "If you were charged twice for the same order, contact billing support with your order ID and the two charge dates. Duplicate charges are refunded automatically within 3-5 business days once verified."},
    {"id": "faq_002", "category": "billing", "question": "Where can I find my invoices?",
     "answer": "Invoices are available under Account Settings > Billing > Invoice History. You can download any invoice as a PDF from there."},
    {"id": "faq_003", "category": "billing", "question": "What payment methods are accepted?",
     "answer": "We accept all major credit cards, PayPal, and SEPA direct debit for customers in the EU."},
    {"id": "faq_004", "category": "billing", "question": "How do I update my billing address?",
     "answer": "Go to Account Settings > Billing > Address, update the fields, and click Save. The new address applies to your next invoice."},
    {"id": "faq_005", "category": "technical_support", "question": "The app keeps crashing, what should I do?",
     "answer": "First, make sure you're on the latest version. If the crash continues, clear the app cache under Settings > Storage, then restart your device. If it still crashes, send us the crash log from Settings > Help > Send Diagnostics."},
    {"id": "faq_006", "category": "technical_support", "question": "How do I reset two-factor authentication?",
     "answer": "Go to Account Settings > Security > Two-Factor Authentication and click Reset. You'll need to verify your identity via your recovery email."},
    {"id": "faq_007", "category": "technical_support", "question": "What are the system requirements?",
     "answer": "The desktop app requires Windows 10+, macOS 12+, or a modern Linux distribution with at least 4GB RAM. The web app works on any browser released in the last 2 years."},
    {"id": "faq_008", "category": "technical_support", "question": "Is there a known issue with Safari?",
     "answer": "Some users on Safari 16 experience a sync delay of up to 60 seconds. This is a known issue and a fix is scheduled for the next release; Chrome and Firefox are unaffected."},
    {"id": "faq_009", "category": "account_login", "question": "I can't log in and never receive the password reset email.",
     "answer": "Check your spam folder first. If it's not there, wait 10 minutes (emails can be delayed) and try again. If you still don't receive it, contact support with your account email so we can manually trigger a reset."},
    {"id": "faq_010", "category": "account_login", "question": "How do I change the email linked to my account?",
     "answer": "Go to Account Settings > Profile > Email, enter the new address, and confirm via the verification link sent to it. Your old email remains active until the new one is verified."},
    {"id": "faq_011", "category": "account_login", "question": "Can I merge two accounts?",
     "answer": "Yes. Contact support with both account emails and we will merge order history and subscriptions into the account you choose as primary within 5 business days."},
    {"id": "faq_012", "category": "cancellation_refund", "question": "What is your refund policy?",
     "answer": "Digital subscriptions can be refunded within 14 days of purchase if unused beyond the trial features. Physical products can be returned within 30 days in original condition for a full refund."},
    {"id": "faq_013", "category": "cancellation_refund", "question": "How do I cancel my subscription?",
     "answer": "Go to Account Settings > Subscription > Cancel Plan. Cancellation takes effect at the end of the current billing period; you keep access until then."},
    {"id": "faq_014", "category": "cancellation_refund", "question": "Can I pause my subscription instead of cancelling?",
     "answer": "Yes, under Account Settings > Subscription > Pause, you can pause for up to 3 months. Billing stops immediately and resumes automatically when the pause ends."},
    {"id": "faq_015", "category": "shipping_delivery", "question": "How long does shipping take?",
     "answer": "Standard shipping takes 3-7 business days domestically and 7-14 business days internationally. Express shipping (available at checkout) takes 1-3 business days domestically."},
    {"id": "faq_016", "category": "shipping_delivery", "question": "My order arrived damaged, what do I do?",
     "answer": "Contact support within 48 hours with photos of the damaged item and packaging. We will send a replacement at no cost or issue a full refund, whichever you prefer."},
    {"id": "faq_017", "category": "shipping_delivery", "question": "Can I change my delivery address after placing an order?",
     "answer": "You can change the delivery address within 1 hour of placing the order via Order History > Edit. After that window, contact support and we'll try to intercept the shipment, though this isn't always possible."},
    {"id": "faq_018", "category": "shipping_delivery", "question": "Do you ship internationally?",
     "answer": "Yes, we ship to over 40 countries. International orders may be subject to customs fees which are the buyer's responsibility."},
    {"id": "faq_019", "category": "product_info", "question": "What's the difference between free and paid tiers?",
     "answer": "The free tier includes core features with usage limits (e.g. 5 projects). Paid tiers unlock unlimited projects, priority support, and advanced integrations. See the pricing page for a full comparison."},
    {"id": "faq_020", "category": "product_info", "question": "Is there a student discount?",
     "answer": "Yes, students with a valid .edu or equivalent institutional email get 50% off any paid plan. Apply under Account Settings > Billing > Student Discount."},
    {"id": "faq_021", "category": "product_info", "question": "Does the product integrate with third-party calendars?",
     "answer": "Yes, we support two-way sync with Google Calendar and Outlook. Connect it under Settings > Integrations > Calendar."},
    {"id": "faq_022", "category": "product_info", "question": "Is the product available in the EU region?",
     "answer": "Yes, and EU customer data is hosted in Frankfurt in compliance with GDPR. You can request a data export or deletion at any time from Account Settings > Privacy."},
]


def build_tickets():
    rows = []
    for intent, by_sentiment in TEMPLATES.items():
        for sentiment, sentences in by_sentiment.items():
            # each base sentence is used with a couple of random slot-fills
            for sentence in sentences:
                n_variants = 4 if "{" in sentence else 1
                for _ in range(n_variants):
                    text = sentence.format(
                        product=random.choice(PRODUCTS),
                        order=random.choice(ORDER_IDS),
                    )
                    rows.append({"text": text, "intent": intent, "sentiment_label": sentiment})
    random.shuffle(rows)
    return rows


def main():
    rows = build_tickets()
    tickets_path = DATA_DIR / "tickets.csv"
    with open(tickets_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "sentiment_label"])
        writer.writeheader()
        writer.writerows(rows)

    kb_path = DATA_DIR / "faq_knowledge_base.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(FAQ_KB, f, indent=2)

    print(f"Wrote {len(rows)} tickets to {tickets_path}")
    print(f"Wrote {len(FAQ_KB)} FAQ articles to {kb_path}")

    # quick class balance report
    from collections import Counter
    intents = Counter(r["intent"] for r in rows)
    sentiments = Counter(r["sentiment_label"] for r in rows)
    print("Intent distribution:", dict(intents))
    print("Sentiment distribution:", dict(sentiments))


if __name__ == "__main__":
    main()
