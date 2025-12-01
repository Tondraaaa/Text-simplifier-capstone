from transformers import pipeline
import re

simplifier = pipeline("text2text-generation", model="eugenesiow/bart-paraphrase")

def simplify(text):
    """Simplifies text by paraphrasing it in plain English."""
    prompt = f"paraphrase: {text}"
    out = simplifier(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )[0]["generated_text"]

    # Remove the 'Paraphrase:' prefix if it appears
    out = re.sub(r"(?i)^paraphrase:\s*", "", out.strip())
    return out.strip()


print("----A----")
print(simplify("Artificial intelligence is transforming libraries and archives by improving discovery, but it may encode bias."))

print("\n----B----")
print(simplify("Quantum error correction protects qubits from decoherence by distributing information redundantly across entangled states."))

