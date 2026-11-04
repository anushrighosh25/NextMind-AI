import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size=50257,
        embedding_dim=64,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, input_ids):

        embedded = self.embedding(input_ids)

        output, _ = self.lstm(embedded)

        logits = self.fc(output)

        return logits