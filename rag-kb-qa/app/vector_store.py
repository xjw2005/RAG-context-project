import faiss
import numpy as np
from typing import List, Dict, Any

class VectorStore:
def __init__(self, dim: int):
self.dim = dim
self.index = faiss.IndexFlatIP(dim) # 余弦相似可用归一化+内积
self.meta: List[Dict[str, Any]] = []

def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
# vectors: [n, dim]
vectors = vectors.astype("float32")
faiss.normalize_L2(vectors)
self.index.add(vectors)
self.meta.extend(metas)

def search(self, query_vec: np.ndarray, top_k: int = 3):
query_vec = query_vec.astype("float32").reshape(1, -1)
faiss.normalize_L2(query_vec)
scores, ids = self.index.search(query_vec, top_k)
results = []
for score, idx in zip(scores[0], ids[0]):
if idx == -1:
continue
results.append({
"score": float(score),
"meta": self.meta[idx]
})
return results

def size(self):
return self.index.ntotal
