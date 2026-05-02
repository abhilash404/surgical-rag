import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, corpus_path="data/corpus.json"):
        with open(corpus_path, "r") as f:
            data = json.load(f)
        
        self.passages = data["passages"]  # list of {id, text}
        self.texts = [p["text"] for p in self.passages]
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Build FAISS index
        embeddings = self.model.encode(self.texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine on normalized
        self.index.add(embeddings)
    
    def retrieve(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "id": self.passages[idx]["id"],
                "text": self.passages[idx]["text"],
                "score": float(score)
            })
        return results


if __name__ == "__main__":
    r = Retriever()
    results = r.retrieve("Who discovered DNA double helix structure?")
    for res in results:
        print(f"{res['id']} ({res['score']:.3f}): {res['text'][:80]}...")