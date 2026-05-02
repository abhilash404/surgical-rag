# Surgical-RAG: A Targeted Correction Framework for Efficient Retrieval-Augmented Generation

## Overview

Surgical-RAG is an optimized Retrieval-Augmented Generation (RAG) framework designed to reduce unnecessary computation during response correction. Traditional RAG systems rely on **global regeneration**, where even minor factual errors trigger full response regeneration.

Surgical-RAG introduces a **targeted correction mechanism** that identifies and repairs only the incorrect segments, significantly improving efficiency.

This repository implements the framework proposed in the paper:
📄 *Surgical-RAG: A Targeted Correction Framework for Efficient Retrieval-Augmented Generation* 

---

## Key Idea

Instead of treating generated text as immutable, Surgical-RAG treats it as an **editable structure**:

* Detect hallucinations at the **claim level**
* Apply **selective correction**
* Preserve correct segments to reduce token usage

---

## Architecture

Pipeline flow:

```
Query → Retriever → Generator → Verifier → Selective Regenerator → Final Output
```

### Components

* **Retriever (`Retriever`)**

  * Retrieves top-k relevant passages
  * Model: `all-MiniLM-L6-v2`

* **Reranker (`Reranker`)**

  * Improves retrieval relevance (optional but included)

* **Generator (`Generator`)**

  * Base LLM generation using Groq (Llama 3.3 70B)

* **Claim Verifier (`ClaimVerifier`)**

  * NLI-based verification using DeBERTa
  * Classifies claims as Supported / Unsupported

* **Selective Regenerator (`SelectiveRegenerator`)**

  * Repairs only incorrect claims using masked regeneration

---

## Core Innovation: Delta-Correction Mask

Instead of regenerating the full response, we apply masking only to incorrect claims:

$$
\mathcal{L}*{repair} = \sum*{m \in \text{Mask}} \log P(T_m | T_{\setminus m}, Q, P)
$$

This ensures:

* Preservation of correct text
* Reduced token consumption
* Lower latency

---

## Methodology

1. Generate baseline response
2. Decompose into claims
3. Verify each claim using NLI
4. Mark unsupported claims
5. Regenerate only masked segments
6. Combine into final response

---

## Performance

### Key Result

* **28.2% average token efficiency gain** over standard RAG 

### Observations

* High efficiency when hallucination rate is low
* Performance degrades beyond ~75% error density
* Suggests need for **adaptive fallback to full regeneration**

---

## Efficiency Insight

* Works best when:

  * Most of the response is correct
* Less effective when:

  * Entire response is incorrect

---

## Installation

```bash
git clone https://github.com/abhilash404/surgical-rag.git
cd surgical-rag
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

Ensure `.env` is added to `.gitignore`.

---

## Usage

```python
pipeline = Pipeline()

result = pipeline.run("Your query here")

print(result)
```

---

## File Structure

```
.
├── retriever.py
├── reranker.py
├── generator.py
├── verifier.py
├── regenerator.py
├── evaluate.py
└── README.md
```

---

## Evaluation

Run:

```bash
python evaluate.py
```

This evaluates:

* Token usage
* Error correction
* Efficiency gains

---

## Limitations

* Depends on verifier accuracy (DeBERTa rigidity)
* May misclassify semantically correct but rephrased answers
* Less efficient at high hallucination density

---

## Future Work

* Adaptive switching between Surgical and Full RAG
* Improved semantic verification (beyond strict NLI)
* Streaming / real-time correction

---

## Acknowledgment

Department of Computer Science and Technology
Silicon University, Bhubaneswar

---

## License

MIT License
