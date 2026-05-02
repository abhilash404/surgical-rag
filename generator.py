```python
import os
from groq import Groq


class Generator:
    def __init__(self):
        # Load API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables")

        # Initialize Groq client
        self.client = Groq(api_key=api_key)

        # Model configuration
        self.model = "llama-3.3-70b-versatile"

    def generate(self, query, passages, force_long=True):
        # Build context string
        context = "\n\n".join([f"[{p['id']}]: {p['text']}" for p in passages])

        # Optional length constraint
        length_instruction = (
            "Provide a detailed 4-sentence paragraph." if force_long else ""
        )

        # Prompt
        prompt = f"""You are a factual question answering assistant.
Answer the question using ONLY the provided context. {length_instruction}
Write in complete sentences.

Context:
{context}

Question: {query}

Answer:"""

        # API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        return (
            response.choices[0].message.content.strip(),
            response.usage.total_tokens,
        )
```
