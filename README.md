# ⚖️ Hybrid Legal Search (Single Docker Demo with Cross-Encoder)

This repository demonstrates **Hybrid Search** for legal texts using a single Docker container.  
It combines:
- **BM25 (keyword search)**
- **Vector embeddings (semantic search, SentenceTransformers)**
- **Hybrid fusion + Cross-Encoder re-ranking**

Includes:
- Sample legal-text dataset
- Interactive frontend (search, α-slider, query highlighting, pagination, history)
- Evaluation metrics (Precision@k, Recall@k, NDCG@k)

## Quick Start
```bash
git clone https://github.com/<your-username>/hybrid-legal-search.git
cd hybrid-legal-search
docker build -t hybrid-legal-search .
docker run --rm -p 8000:8000 hybrid-legal-search
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

## Evaluation
```bash
docker run --rm hybrid-legal-search python -c "from app.evaluate import HybridSearch, evaluate; e=HybridSearch(); evaluate(e, k=5)"
```

Cross-Encoder re-ranking improves **Precision@5** and **NDCG@5** for top results.
