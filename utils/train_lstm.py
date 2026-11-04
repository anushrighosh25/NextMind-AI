import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from lstm_model import LSTMLanguageModel


# 1. Select CPU
device = torch.device("cpu")

# 2. Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 3. Small training dataset
training_text = """
Artificial intelligence helps computers learn from data.
Machine learning is a branch of artificial intelligence.
Deep learning uses neural networks to learn complex patterns.
Natural language processing helps computers understand human language.
Language models predict the next token in a sequence.
GPT models use transformer architectures.
LSTM networks process sequential information.
Python is widely used for machine learning and artificial intelligence.
"""

# 4. Convert text into token IDs
token_ids = tokenizer.encode(training_text)

# 5. Prepare input and target sequences
sequence_length = 16

inputs = []
targets = []

for i in range(len(token_ids) - sequence_length):
    inputs.append(token_ids[i:i + sequence_length])
    targets.append(token_ids[i + 1:i + sequence_length + 1])


# 6. Create dataset
class TextDataset(Dataset):

    def __len__(self):
        return len(inputs)

    def __getitem__(self, index):
        return (
            torch.tensor(inputs[index]),
            torch.tensor(targets[index])
        )


dataset = TextDataset()

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# 7. Initialize LSTM
model = LSTMLanguageModel(
    vocab_size=tokenizer.vocab_size
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 8. Train model
epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output.reshape(-1, tokenizer.vocab_size),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Loss: {average_loss:.4f}"
    )


# 9. Save trained model
torch.save(
    model.state_dict(),
    "utils/lstm_model.pth"
)

print("\nLSTM training completed!")
print("Model saved successfully.")