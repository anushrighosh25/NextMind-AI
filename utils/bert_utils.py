import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM


@st.cache_resource
def load_bert():

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


def get_masked_word_predictions(sentence, top_k=5):

    tokenizer, model = load_bert()

    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )

    mask_token_id = tokenizer.mask_token_id

    mask_positions = (
        inputs["input_ids"] == mask_token_id
    ).nonzero(as_tuple=True)[1]

    if len(mask_positions) != 1:
        raise ValueError(
            "Please include exactly one [MASK] token."
        )

    mask_position = mask_positions.item()

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits[0, mask_position, :]

    probabilities = torch.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(
        probabilities,
        top_k
    )

    results = []

    for probability, token_id in zip(top_probs, top_ids):

        token = tokenizer.decode(
            [token_id.item()]
        )

        results.append({
            "token": token,
            "probability": probability.item()
        })

    return results