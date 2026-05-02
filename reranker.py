from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, threshold=0.3):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.threshold = threshold
    
    def rerank(self, query, passages):
        # passages = list of {id, text, score} from retriever
        pairs = [[query, p["text"]] for p in passages]
        scores = self.model.predict(pairs)
        
        # Attach reranker scores
        for i, passage in enumerate(passages):
            passage["rerank_score"] = float(scores[i])
        
        # Sort by reranker score
        reranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
        
        # Filter below threshold
        filtered = [p for p in reranked if p["rerank_score"] > self.threshold]
        
        # Always keep at least 1 passage, but only if score is positive
        if not filtered:
            if reranked[0]["rerank_score"] > 0:
                filtered = [reranked[0]]
            else:
                # No relevant passage found — return empty to signal this
                filtered = []
        
        return filtered


if __name__ == "__main__":
    from retriever import Retriever
    
    r = Retriever()
    reranker = Reranker()
    
    query = "Who discovered DNA double helix structure?"
    retrieved = r.retrieve(query, top_k=5)
    
    print("Before reranking:")
    for p in retrieved:
        print(f"  {p['id']} ({p['score']:.3f}): {p['text'][:60]}...")
    
    filtered = reranker.rerank(query, retrieved)
    
    print(f"\nAfter reranking + filtering (threshold=0.3), {len(filtered)} passages kept:")
    for p in filtered:
        print(f"  {p['id']} (rerank={p['rerank_score']:.3f}): {p['text'][:60]}...")