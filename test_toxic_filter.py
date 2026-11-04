import re
from utils.sampling_utils import generate_text

prompt = "In a small garden"
blocked_words = ["the"]

print("U5 - Toxic Token Filtering Test")
print("--------------------------------")

generated = generate_text(
    prompt=prompt,
    max_new_tokens=40,
    temperature=0.8,
    top_k=50,
    blocked_words=blocked_words
)

print("\nGenerated Text:")
print(generated)

# Check for the complete blocked word
found = any(
    re.search(
        r"\b" + re.escape(word) + r"\b",
        generated,
        re.IGNORECASE
    )
    for word in blocked_words
)

if found:
    print("\nFAIL: A blocked word was generated.")
else:
    print("\nPASS: No blocked word found!")