import re

class SelectiveRegenerator:
    def __init__(self):
        from generator import Generator
        self.generator = Generator()

    def regenerate(self, original_answer, verification_results, passages):
        broken_indices = [i for i, v in enumerate(verification_results) if v["status"] != "supported"]
        if not broken_indices:
            return original_answer, 0

        # Create a tagged version to guide the model's attention
        tagged_answer = ""
        for i, v in enumerate(verification_results):
            if i in broken_indices:
                tagged_answer += f" [FIX THIS: {v['claim']}] "
            else:
                tagged_answer += f" {v['claim']} "

        context_text = "\n".join([p["text"] for p in passages])

        prompt = (
            f"CONTEXT:\n{context_text}\n\n"
            f"TEXT TO REPAIR: {tagged_answer}\n\n"
            "INSTRUCTION: Rewrite the text. Keep correct sentences unchanged. "
            "Replace [FIX THIS: ...] tags with factual info from context. "
            "If info is missing, use the phrase 'Data unavailable.' "
            "OUTPUT THE FULL PARAGRAPH ONLY."
        )

        final_answer, tokens = self.generator.generate(prompt, passages=[])
        return final_answer.strip(), tokens