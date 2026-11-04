
import torch
from utils.sampling_utils import generate_text

# Factual questions and expected answers
test_cases = [
    {
        "prompt": "The capital of France is",
        "expected": "Paris"
    },
    {
        "prompt": "The chemical symbol for gold is",
        "expected": "Au"
    },
    {
        "prompt": "The closest planet to the Sun is",
        "expected": "Mercury"
    },
    {
        "prompt": "Water freezes at",
        "expected": "0 degrees Celsius"
    },
    {
        "prompt": "The largest planet in our solar system is",
        "expected": "Jupiter"
    },
    {
        "prompt": "The capital of Japan is",
        "expected": "Tokyo"
    },
    {
        "prompt": "The Red Planet is",
        "expected": "Mars"
    },
    {
        "prompt": "The square root of 81 is",
        "expected": "9"
    },
    {
        "prompt": "The author of Romeo and Juliet is",
        "expected": "William Shakespeare"
    },
    {
        "prompt": "The primary language spoken in Brazil is",
        "expected": "Portuguese"
    }
]

temperatures = [0.2, 1.2]

print("=" * 55)
print("U5 - FACTUAL RESPONSE EVALUATION")
print("=" * 55)

results = {}

for temp in temperatures:

    print(f"\n========== TEMPERATURE {temp} ==========")

    for index, case in enumerate(test_cases):

        # Reset seed for reproducible sampling
        torch.manual_seed(42 + index)

        output = generate_text(
            prompt=case["prompt"],
            max_new_tokens=20,
            temperature=temp,
            top_k=50
        )

        # Extract generated continuation
        continuation = output[len(case["prompt"]):].strip()

        print("\nQuestion:", case["prompt"])
        print("Expected:", case["expected"])
        print("Generated:", continuation)

    print("\nEnd of temperature", temp, "evaluation.")

print("\n" + "=" * 55)
print("EVALUATION COMPLETED")
print("=" * 55)

print("\nIMPORTANT:")
print("Review each generated answer manually.")
print("Do not treat keyword presence as proof of factual correctness.")
