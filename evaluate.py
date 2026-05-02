import json
from retriever import Retriever
from claim_verifier import ClaimVerifier
from selective_regen import SelectiveRegenerator
from generator import Generator 

def verify_answer(verifier, answer, passages):
    if not passages or not answer: return [], 0, 0
    verification = verifier.verify_answer(str(answer), passages)
    flagged = [v for v in verification if v["status"] in ("unsupported", "contradicted")]
    return verification, len(flagged), len(verification)

def run_evaluation():
    print("Loading Research Suite...")
    retriever, verifier, regen, generator = Retriever(), ClaimVerifier(), SelectiveRegenerator(), Generator() 
    
    test_cases = [
        {"query": "What is the molecular structure of DNA?", "baseline_answer": "DNA is a molecule that carries genetic instructions. It consists of two polynucleotide chains that coil around each other. These chains are composed of smaller units called nucleotides. The structure is a triple helix discovered by Marie Curie in 1945.", "passage_id": "p5"},
        {"query": "Who were the astronauts in Apollo 11?", "baseline_answer": "The Apollo 11 mission was the first to land humans on a celestial body. It was launched from Kennedy Space Center. The crew included Neil Armstrong and Buzz Aldrin. They landed their spacecraft on Mars in 1972.", "passage_id": "p4"},
        {"query": "Which book did Isaac Newton publish in 1687?", "baseline_answer": "Isaac Newton was an English mathematician and physicist. He is widely recognized as one of the greatest scientists of all time. In 1687, he established the laws of motion. He published his findings in 'On the Origin of Species'.", "passage_id": "p3"},
        {"query": "In which scientific fields did Marie Curie win Nobel Prizes?", "baseline_answer": "Marie Curie was a pioneering physicist and chemist. She conducted pioneering research on radioactivity. She won a Nobel Prize in Physics in 1903. She won her second Nobel Prize in Economics in 1911.", "passage_id": "p9"},
        {"query": "How many people died in World War II?", "baseline_answer": "World War II was a global conflict that lasted from 1939 to 1945. It involved the vast majority of the world's countries. The war involved the mobilization of over 100 million personnel. Total fatalities are estimated at 2 million people.", "passage_id": "p11"}
    ]

    header = f"{'QUERY':<25} {'BASE_E':>6} {'FULL_E':>6} {'SURG_E':>6} {'F_GEN':>8} {'S_GEN':>8} {'SAVINGS':>8}"
    print("\n" + "="*len(header) + "\n" + header + "\n" + "="*len(header))

    stats = {"b_err": 0, "f_err": 0, "s_err": 0, "claims": 0, "f_cost": 0, "s_cost": 0}

    for tc in test_cases:
        passages = retriever.retrieve(tc["query"], top_k=5)
        passages = [p for p in passages if p["id"] == tc["passage_id"]] or passages[:1]

        # 1. Baseline Verification
        b_ver, b_err, b_count = verify_answer(verifier, tc["baseline_answer"], passages)
        
        # 2. Full Regeneration
        f_ans, _ = generator.generate(tc["query"], passages, force_long=True)
        _, f_err, _ = verify_answer(verifier, f_ans, passages)

        # 3. Surgical Fix
        s_ans, _ = regen.regenerate(tc["baseline_answer"], b_ver, passages)
        _, s_err, _ = verify_answer(verifier, s_ans, passages)

        # MATH: Delta Cost Logic
        f_words = len(f_ans.split())
        s_words_total = len(s_ans.split())
        # Surgical only 'pays' for the portion it attempted to fix
        s_gen_cost = s_words_total * (b_err / b_count) 

        savings = ((f_words - s_gen_cost) / f_words * 100) if f_words > 0 else 0

        # Accumulate Stats
        stats["b_err"] += b_err
        stats["f_err"] += f_err
        stats["s_err"] += s_err
        stats["claims"] += b_count
        stats["f_cost"] += f_words
        stats["s_cost"] += s_gen_cost

        print(f"{tc['query'][:25]:<25} {b_err:>6} {f_err:>6} {s_err:>6} {f_words:>8.0f} {s_gen_cost:>8.0f} {savings:>7.1f}%")

    print("=" * len(header))
    
    # Summary Calculations
    b_rate = (stats["b_err"] / stats["claims"]) * 100
    f_rate = (stats["f_err"] / stats["claims"]) * 100
    s_rate = (stats["s_err"] / stats["claims"]) * 100
    total_sav = (stats["f_cost"] - stats["s_cost"]) / stats["f_cost"] * 100

    print("\n--- FINAL COMPARATIVE RESEARCH SUMMARY ---")
    print(f"1. Baseline Error Rate:    {b_rate:.1f}%")
    print(f"2. Full Regen Error Rate:  {f_rate:.1f}%")
    print(f"3. Surgical Error Rate:    {s_rate:.1f}%")
    print("-" * 55)
    print(f"Accuracy Improvement (vs Baseline):   {(b_rate - s_rate):.1f} points")
    print(f"NET GENERATION EFFICIENCY GAIN:       {total_sav:.1f}%")
    print("-" * 55)

if __name__ == "__main__":
    run_evaluation()