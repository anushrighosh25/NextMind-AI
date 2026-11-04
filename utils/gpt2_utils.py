
import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    attn_implementation="eager"
)
    model.eval()
    return tokenizer, model


def get_next_token_probabilities(
    prompt,
    top_k=5,
    temperature=1.0,
    top_p=1.0
):
    tokenizer, model = load_model()

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Get next-token logits
    logits = outputs.logits[0, -1, :].clone()

    # Apply temperature
    logits = logits / temperature

    # Apply Top-K filtering
    if top_k > 0:
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float("-inf")

    # Apply Top-P filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True
        )

        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_tokens = cumulative_probs > top_p

        # Keep the token that crosses the threshold
        remove_tokens[1:] = remove_tokens[:-1].clone()
        remove_tokens[0] = False

        indices_to_remove = sorted_indices[remove_tokens]
        logits[indices_to_remove] = float("-inf")

    # Calculate probabilities after filtering
    probabilities = torch.softmax(logits, dim=-1)

    # Return the remaining top predictions
    display_count = min(top_k, int(torch.isfinite(logits).sum()))
    top_probs, top_ids = torch.topk(probabilities, display_count)

    results = []

    for probability, token_id in zip(top_probs, top_ids):
        results.append({
            "token": tokenizer.decode([token_id.item()]),
            "probability": probability.item()
        })

    return results
def get_attention_matrix(prompt, layer=-1, head=0):

    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True
        )

    attention = outputs.attentions[layer][0, head]

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    return tokens, attention.cpu().numpy()
def get_attention_matrix(prompt, layer=-1, head=0):

    tokenizer, model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True
        )

    attention = outputs.attentions[layer][0, head]

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    return tokens, attention.cpu().numpy()