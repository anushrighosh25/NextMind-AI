
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("Loading GPT-2 tokenizer...")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Loading GPT-2 model...")

# Load the pretrained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Use CPU
model.eval()

print("\nGPT-2 loaded successfully!")

# Test the model
prompt = "Artificial Intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get probabilities for the next token
next_token_logits = outputs.logits[0, -1, :]
probabilities = torch.softmax(next_token_logits, dim=-1)

# Find the top 5 predictions
top_probs, top_ids = torch.topk(probabilities, 5)

print("\nPrompt:", prompt)
print("\nTop 5 next-token predictions:")

for probability, token_id in zip(top_probs, top_ids):
    token = tokenizer.decode([token_id.item()])
    print(f"{token!r}: {probability.item():.4f}")
