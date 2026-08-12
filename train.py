"""
One-time setup script: generates data (if missing), trains the intent classifier,
and builds the FAISS retrieval index. Run this once before app.py or the tests.

    python train.py
"""
from pathlib import Path
from src.intent_classifier import train_intent_classifier
from src.retriever import build_index

ROOT = Path(__file__).parent
TICKETS_CSV = ROOT / "data" / "tickets.csv"
KB_JSON = ROOT / "data" / "faq_knowledge_base.json"


def main():
    if not TICKETS_CSV.exists() or not KB_JSON.exists():
        print("Data files missing, generating synthetic dataset...")
        import subprocess
        subprocess.run(["python3", str(ROOT / "data" / "generate_data.py")], check=True)

    print("\n--- Training intent classifier ---")
    result = train_intent_classifier(str(TICKETS_CSV))
    print(f"Test accuracy: {result['accuracy']:.3f}")
    print(result["report"])

    print("--- Building retrieval index ---")
    build_index(str(KB_JSON))
    print("Index built and saved to models/")

    print("\nSetup complete. Run `streamlit run app.py` or `python3 -m src.pipeline` to try it.")


if __name__ == "__main__":
    main()
