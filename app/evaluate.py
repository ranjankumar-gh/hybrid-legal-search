import numpy as np
from app.dataset import queries
from app.search_engine import HybridSearch

def precision_at_k(results, relevant, k):
    retrieved = [r["id"] for r in results[:k]]
    return len(set(retrieved) & set(relevant)) / k

def recall_at_k(results, relevant, k):
    retrieved = [r["id"] for r in results[:k]]
    return len(set(retrieved) & set(relevant)) / len(relevant)

def ndcg_at_k(results, relevant, k):
    dcg = 0.0
    for i, r in enumerate(results[:k]):
        if r["id"] in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate(engine=None, k=5):
    if engine is None:
        engine = HybridSearch()

    metrics = {"precision": [], "recall": [], "ndcg": []}
    for q, relevant in queries.items():
        res = engine.search(q, alpha=0.5, top_k=k)
        metrics["precision"].append(precision_at_k(res, relevant, k))
        metrics["recall"].append(recall_at_k(res, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(res, relevant, k))

    print("Precision@%d: %.3f" % (k, np.mean(metrics["precision"])))
    print("Recall@%d: %.3f" % (k, np.mean(metrics["recall"])))
    print("NDCG@%d: %.3f" % (k, np.mean(metrics["ndcg"])))
