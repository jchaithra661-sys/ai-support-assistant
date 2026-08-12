"""
Sentiment scoring via VADER (lexicon + rule-based, tuned for short informal text
like tickets/chat — a good fit here and doesn't require downloading a model file).

Why not a transformer sentiment model? For a Werkstudent-level demo, it's more
convincing to show you understand the tradeoff (speed/interpretability vs. accuracy
on nuanced text) than to import a black box. The "Level up" section in README.md
shows the one-line swap to a HuggingFace sentiment pipeline if you want to extend this.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": round(compound, 3), "scores": scores}


if __name__ == "__main__":
    examples = [
        "I was charged twice and nobody has refunded me yet, this is unacceptable.",
        "Can you explain the charge on my statement?",
        "Thanks for sorting out my refund so quickly!",
    ]
    for ex in examples:
        print(ex, "->", analyze_sentiment(ex))
