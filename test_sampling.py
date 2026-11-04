from utils.sampling_utils import generate_text

prompt = "Artificial intelligence will"

print("\nGPT-2 Sampling Test")
print("-------------------")

generated = generate_text(
    prompt,
    max_new_tokens=30,
    temperature=1.0,
    top_k=50
)

print("\nGenerated Text:")
print(generated)