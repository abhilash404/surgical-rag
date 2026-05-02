from transformers import pipeline

class ClaimVerifier:
    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1  # CPU, change to 0 if GPU works
        )
    
    def extract_claims(self, answer):
        # Split answer into individual sentences = individual claims
        import re
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def verify_claim(self, claim, passages):
        labels_found = []

        for passage in passages:
            result = self.nli(
                passage['text'],
                text_pair=claim
            )[0]
            labels_found.append({
                "passage_id": passage["id"],
                "label": result["label"],
                "score": result["score"]
            })

        for r in labels_found:
            if r["label"] == "entailment":        # lowercase
                return "supported", labels_found

        for r in labels_found:
            if r["label"] == "contradiction":     # lowercase
                return "contradicted", labels_found

        return "unsupported", labels_found

        
    def verify_answer(self, answer, passages):
        claims = self.extract_claims(answer)
        results = []
        
        for claim in claims:
            status, details = self.verify_claim(claim, passages)
            results.append({
                "claim": claim,
                "status": status,  # supported / unsupported / contradicted
                "details": details
            })
        
        return results


if __name__ == "__main__":
    from retriever import Retriever
    from reranker import Reranker
    from generator import Generator

    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()
    verifier = ClaimVerifier()

    query = "Who discovered DNA double helix structure?"
    retrieved = retriever.retrieve(query, top_k=5)
    filtered = reranker.rerank(query, retrieved)
    answer = generator.generate(query, filtered)

    print(f"Answer: {answer}\n")

    verification = verifier.verify_answer(answer, filtered)
    for v in verification:
        print(f"Claim: {v['claim']}")
        print(f"Status: {v['status']}")
        print()