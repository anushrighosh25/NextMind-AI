import torch
import torch.nn as nn
from utils.lstm_model import LSTMLanguageModel
from utils.gpt2_utils import load_model


# Use CPU
device = torch.device("cpu")

# Load GPT-2 tokenizer and model
tokenizer, gpt2_model = load_model()

# Test sentence (not copied directly from training data)
test_text = (
    "Neural networks can generate meaningful text "
    "by learning patterns from language data."
)

# Tokenize the same sentence for both models
input_ids = tokenizer(
    test_text,
    return_tensors="pt"
)["input_ids"].to(device)

# Cross-entropy loss
criterion = nn.CrossEntropyLoss()

# --------------------------------
# 1. Calculate GPT-2 Perplexity
# --------------------------------

gpt2_model.eval()

with torch.no_grad():

    gpt2_outputs = gpt2_model(input_ids)

    gpt2_logits = gpt2_outputs.logits[:, :-1, :]

    targets = input_ids[:, 1:]

    gpt2_loss = criterion(
        gpt2_logits.reshape(-1, tokenizer.vocab_size),
        targets.reshape(-1)
    )

    gpt2_perplexity = torch.exp(gpt2_loss).item()


# --------------------------------
# 2. Calculate LSTM Perplexity
# --------------------------------

lstm_model = LSTMLanguageModel(
    vocab_size=tokenizer.vocab_size
).to(device)

lstm_model.load_state_dict(
    torch.load(
        "utils/lstm_model.pth",
        map_location=device
    )
)

lstm_model.eval()

with torch.no_grad():

    lstm_logits = lstm_model(input_ids[:, :-1])

    lstm_loss = criterion(
        lstm_logits.reshape(-1, tokenizer.vocab_size),
        targets.reshape(-1)
    )

    lstm_perplexity = torch.exp(lstm_loss).item()


# --------------------------------
# 3. Display Results
# --------------------------------

print("\nPERPLEXITY COMPARISON")
print("-----------------------------")

print(f"Test sentence: {test_text}")

print(f"\nGPT-2 Perplexity: {gpt2_perplexity:.2f}")

print(f"LSTM Perplexity: {lstm_perplexity:.2f}")

print("\nComparison completed successfully!")