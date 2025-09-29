from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from app.dataset import documents

class HybridSearch:
    def __init__(self):
        self.docs = [d["text"] for d in documents]
        self.doc_ids = [d["id"] for d in documents]
        self.tokenized_docs = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = self.model.encode(self.docs, normalize_embeddings=True)

    def search(self, query, alpha=0.5, top_k=5):
        # BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-9)

        # Vector
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        vec_scores = np.dot(self.doc_embeddings, q_emb)
        vec_scores = (vec_scores - np.min(vec_scores)) / (np.ptp(vec_scores) + 1e-9)

        # Hybrid
        scores = alpha * vec_scores + (1 - alpha) * bm25_scores
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [{"id": self.doc_ids[i], "text": self.docs[i], "score": float(scores[i])} for i in top_idx]
        return results
