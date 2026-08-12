"""
RAG retrieval layer: TF-IDF -> TruncatedSVD (i.e. LSA) dense vectors -> FAISS
similarity search.

Why LSA instead of a neural embedding model (e.g. sentence-transformers)?
Same reasoning as sentiment.py: this ships fully offline with no multi-hundred-MB
model download, which matters if you're demoing this on a laptop with no GPU or
in a locked-down environment (exactly what happened building this — see README).
It is a real, well-established technique (LSA/LSI), not a toy shortcut, and the
FAISS index / retrieval interface below is IDENTICAL to what you'd use with real
neural embeddings — see README "Level up" for the one-function swap to
sentence-transformers if you want stronger semantic matching for your final
submission.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
VECTORIZER_PATH = MODELS_DIR / "retriever_tfidf.joblib"
SVD_PATH = MODELS_DIR / "retriever_svd.joblib"
INDEX_PATH = MODELS_DIR / "retriever.faiss"
DOCS_PATH = MODELS_DIR / "retriever_docs.json"


def load_kb(kb_path: str) -> list:
    with open(kb_path, encoding="utf-8") as f:
        return json.load(f)


def _doc_text(article: dict) -> str:
    # embed question+answer so both a question-style and a keyword-style query can match
    return f"{article['question']} {article['answer']}"


def build_index(kb_path: str, n_components: int = 100, save: bool = True):
    kb = load_kb(kb_path)
    texts = [_doc_text(a) for a in kb]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_comp = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    dense = svd.fit_transform(tfidf_matrix).astype("float32")

    # normalize so inner product == cosine similarity
    faiss.normalize_L2(dense)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    if save:
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(svd, SVD_PATH)
        faiss.write_index(index, str(INDEX_PATH))
        with open(DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(kb, f)

    return {"vectorizer": vectorizer, "svd": svd, "index": index, "docs": kb}


def load_index():
    vectorizer = joblib.load(VECTORIZER_PATH)
    svd = joblib.load(SVD_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    return {"vectorizer": vectorizer, "svd": svd, "index": index, "docs": docs}


def retrieve(query: str, resources: dict, top_k: int = 3) -> list:
    vectorizer, svd, index, docs = (
        resources["vectorizer"], resources["svd"], resources["index"], resources["docs"]
    )
    q_tfidf = vectorizer.transform([query])
    q_dense = svd.transform(q_tfidf).astype("float32")
    faiss.normalize_L2(q_dense)

    scores, idxs = index.search(q_dense, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        doc = docs[idx]
        results.append({**doc, "similarity": round(float(score), 3)})
    return results


if __name__ == "__main__":
    kb_path = str(Path(__file__).parent.parent / "data" / "faq_knowledge_base.json")
    resources = build_index(kb_path)
    for q in ["I got charged twice for my order", "app crashes on startup", "how fast is shipping"]:
        print(f"\nQuery: {q}")
        for r in retrieve(q, resources, top_k=2):
            print(f"  [{r['similarity']}] {r['question']}")
