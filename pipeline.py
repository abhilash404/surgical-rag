from retriever import Retriever
from reranker import Reranker
from generator import Generator
from claim_verifier import ClaimVerifier
from selective_regen import SelectiveRegenerator

class AdaptiveRAGPipeline:
    def __init__(self):
        print("Loading models...")
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()
        self.verifier = ClaimVerifier()
        self.regenerator = SelectiveRegenerator()
        print("Ready.\n")
    
    def run(self, query, top_k=5, verbose=True):
        if verbose:
            print(f"Query: {query}")
            print("-" * 60)
        
        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        if verbose:
            print(f"Retrieved {len(retrieved)} passages")
        
        # Step 2: Rerank + Filter
        filtered = self.reranker.rerank(query, retrieved)
        if verbose:
            print(f"After filtering: {len(filtered)} passages kept")
            for p in filtered:
                print(f"  [{p['id']}] rerank_score={p['rerank_score']:.3f}")
        
        # Step 3: Generate
        answer = self.generator.generate(query, filtered)
        if verbose:
            print(f"\nGenerated answer: {answer}")
        
        # Step 4: Verify claims
        verification = self.verifier.verify_answer(answer, filtered)
        if verbose:
            print("\nClaim verification:")
            for v in verification:
                print(f"  [{v['status'].upper()}] {v['claim'][:70]}")
        
        # Step 5: Selective regeneration
        flagged = [v for v in verification if v["status"] in ("unsupported", "contradicted")]
        if flagged:
            final_answer, _ = self.regenerator.regenerate(answer, verification, filtered)
            if verbose:
                print(f"\nFinal answer (after selective regen): {final_answer}")
        else:
            final_answer = answer
            if verbose:
                print("\nNo hallucinations detected. Answer unchanged.")
        
        return {
            "query": query,
            "retrieved_count": len(retrieved),
            "filtered_count": len(filtered),
            "original_answer": answer,
            "verification": verification,
            "final_answer": final_answer,
            "hallucinations_found": len(flagged)
        }


if __name__ == "__main__":
    pipeline = AdaptiveRAGPipeline()
    
    test_queries = [
        "Who discovered the DNA double helix structure?",
        "Where is the Eiffel Tower and who designed it?",
        "What did Einstein win the Nobel Prize for?",
        "How long is the Great Wall of China?",
        "Who wrote Romeo and Juliet?",
    ]
    
    for query in test_queries:
        result = pipeline.run(query)
        print("\n" + "=" * 60 + "\n")