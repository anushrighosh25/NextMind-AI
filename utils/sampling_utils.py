
import torch
from utils.gpt2_utils import load_model


def build_blocked_sequences(tokenizer, blocked_words):
    """Convert blocked words into complete GPT-2 token sequences."""

    blocked_sequences = set()

    for word in blocked_words or []:
        word = word.strip()

        if not word:
            continue

        # Check common capitalization and spacing variants.
        variants = {
            word,
            word.lower(),
            word.capitalize(),
            word.upper()
        }

        for variant in variants:
            for text in [variant, " " + variant]:
                token_ids = tokenizer.encode(
                    text,
                    add_special_tokens=False
                )

                if token_ids:
                    blocked_sequences.add(tuple(token_ids))

    return list(blocked_sequences)


def generate_text(
    prompt,
    max_new_tokens=30,
    temperature=1.0,
    top_k=50,
    blocked_words=None
):

    tokenizer, model = load_model()

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if top_k < 1 or top_k > tokenizer.vocab_size:
        raise ValueError("Invalid Top-K value.")

    model.eval()

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"]

    blocked_sequences = build_blocked_sequences(
        tokenizer,
        blocked_words
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            outputs = model(input_ids)

            logits = outputs.logits[:, -1, :].clone()

            # Apply temperature
            logits = logits / temperature

            # Block complete token sequences.
            for sequence in blocked_sequences:

                sequence_length = len(sequence)

                if sequence_length == 1:
                    logits[:, sequence[-1]] = float("-inf")

                elif input_ids.shape[1] >= sequence_length - 1:

                    previous_tokens = input_ids[
                        0, -(sequence_length - 1):
                    ].tolist()

                    if previous_tokens == list(sequence[:-1]):
                        logits[:, sequence[-1]] = float("-inf")

            # Apply Top-K filtering
            top_values, _ = torch.topk(logits, top_k)

            threshold = top_values[:, -1].unsqueeze(-1)

            logits[logits < threshold] = float("-inf")

            # Convert logits into probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=-1
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True
    )

    return generated_text
