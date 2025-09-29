from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.search_engine import HybridSearch
import os

app = FastAPI()
engine = HybridSearch()

# serve frontend
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "frontend", "index.html")) as f:
        return f.read()

@app.get("/search")
def search(q: str, alpha: float = Query(0.5), k: int = Query(5)):
    return {"results": engine.search(q, alpha=alpha, top_k=k)}
