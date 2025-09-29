# ⚖️ Hybrid Legal Search (Single Docker Demo)

This repository demonstrates **Hybrid Search** for legal texts using a single Docker container.  
It combines:
- **BM25 (keyword search)**
- **Vector embeddings (semantic search, SentenceTransformers)**
- **Hybrid fusion** with adjustable weight (α)

Includes:
- A small sample legal-text dataset
- Interactive frontend (search, α-slider, query highlighting, pagination, history)
- Evaluation metrics (Precision@k, Recall@k, NDCG@k)

---

## 🚀 Quick Start

### 1. Clone & Build
```bash
git clone https://github.com/<your-username>/hybrid-legal-search.git
cd hybrid-legal-search
docker build -t hybrid-legal-search .
```

### 2. Run
```bash
docker run --rm -p 8000:8000 hybrid-legal-search
```

Now open 👉 [http://localhost:8000](http://localhost:8000)

---

## 📊 Evaluation

Run evaluation metrics (Precision@5, Recall@5, NDCG@5):

```bash
docker run --rm hybrid-legal-search   python -c "from app.evaluate import HybridSearch, evaluate; e=HybridSearch(); evaluate(e, k=5)"
```

Expected outcome:  
- Keyword search is strong for **exact statutory terms**  
- Vector search is better for **semantic/paraphrastic queries**  
- Hybrid improves **overall ranking (NDCG)**

---

## 🛠️ Repo Structure

```
app/
  dataset.py       # Sample docs + relevance labels
  search_engine.py # HybridSearch class (BM25 + Qdrant + Embeddings)
  main.py          # FastAPI backend (serves API + frontend)
  evaluate.py      # Metrics: precision/recall/NDCG
  frontend/index.html # Rich UI (α slider, pagination, history)
```

---

## 📝 Roadmap

- Add larger dataset integration (e.g., Caselaw, Indian Kanoon)
- Add **cross-encoder reranker** for improved precision
- Deploy to cloud (Qdrant Cloud + HuggingFace Spaces)

---

## 📜 License

MIT
