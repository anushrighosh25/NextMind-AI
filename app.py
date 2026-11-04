
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.gpt2_utils import get_attention_matrix
from utils.gpt2_utils import get_next_token_probabilities
import torch
import torch.nn as nn
from utils.sampling_utils import generate_text

from utils.lstm_model import LSTMLanguageModel
from utils.gpt2_utils import load_model
from utils.bert_utils import get_masked_word_predictions

# Page configuration
st.set_page_config(
    page_title="NextMind AI",
    page_icon="🧠",
    layout="wide"
)

# Application heading
st.title("🧠 NextMind AI")

st.write(
    "Explore how GPT-2 predicts the next token "
    "and how temperature changes its predictions."
)

st.divider()

# User input
st.subheader("1. Enter your prompt")

prompt = st.text_input(
    "Type a sentence:",
    value="Artificial Intelligence is"
)

# Settings
st.subheader("2. Adjust model settings")
col1, col2, col3 = st.columns(3)

with col1:
    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=50,
        value=5,
        key="top_k_slider"
    )

with col2:
    temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="temperature_slider"
    )

with col3:
    top_p = st.slider(
        "Top-P",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key="top_p_slider"
    )

# Generate predictions
if st.button("Predict Next Tokens", type="primary"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")

    else:
        with st.spinner("GPT-2 is generating predictions..."):

            try:
                results = get_next_token_probabilities(
    prompt,
    top_k=top_k,
    temperature=temperature,
    top_p=top_p
)

                st.success("Predictions generated successfully!")

                # Convert results into a DataFrame
                df = pd.DataFrame(results)

                df["Probability (%)"] = (
                    df["probability"] * 100
                )

                # Display predictions
                st.subheader("3. Next-token predictions")

                st.dataframe(
                    df[["token", "Probability (%)"]],
                    use_container_width=True,
                    hide_index=True
                )

                # Display bar chart
                st.subheader("4. Probability visualization")

                chart_data = df.set_index("token")[
                    ["Probability (%)"]
                ]

                st.bar_chart(chart_data)

                st.caption(
                    "Probabilities are calculated over GPT-2's "
                    "full vocabulary before displaying the top-K tokens."
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.divider()

st.header("5. Prompt Engineering Lab")

st.write(
    "Compare how GPT-2 responds to zero-shot "
    "and few-shot prompts."
)

zero_shot_prompt = """
Classify the sentiment:
The movie was amazing.
Sentiment:
"""

few_shot_prompt = """
Review: The food was delicious.
Sentiment: Positive

Review: The service was terrible.
Sentiment: Negative

Review: The movie was amazing.
Sentiment:
"""

if st.button("Compare Prompting Techniques"):

    with st.spinner("Comparing prompts..."):

        zero_results = get_next_token_probabilities(
            zero_shot_prompt,
            top_k=5,
            temperature=1.0,
            top_p=1.0
        )

        few_results = get_next_token_probabilities(
            few_shot_prompt,
            top_k=5,
            temperature=1.0,
            top_p=1.0
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Zero-shot Prompt")
            for item in zero_results:
                st.write(
                    f"{item['token']} : "
                    f"{item['probability']:.4f}"
                )

        with col2:
            st.subheader("Few-shot Prompt")
            for item in few_results:
                st.write(
                    f"{item['token']} : "
                    f"{item['probability']:.4f}"
                )
st.divider()

st.header("6. Generative AI Lifecycle")

st.write(
    "Explore the stages involved in building "
    "and using a Generative AI language model."
)

lifecycle = [
    ("1. Data Collection",
     "Collect text data from different sources."),

    ("2. Data Preprocessing",
     "Clean the text and convert it into tokens."),

    ("3. Model Training",
     "GPT-2 learns patterns and relationships between tokens."),

    ("4. Prompt Input",
     "The user provides an input prompt."),

    ("5. Probability Calculation",
     "GPT-2 calculates probabilities for possible next tokens."),

    ("6. Text Generation",
     "The model selects tokens and generates text.")
]

for i, (stage, description) in enumerate(lifecycle):

    st.subheader(stage)
    st.write(description)

    if i != len(lifecycle) - 1:
        st.markdown("⬇️")
        st.divider()

st.header("7. Generative vs Discriminative AI")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Generative AI")
    st.write("Learns patterns in data and generates new content.")
    st.success("Example: GPT-2 generates the next token.")

with col2:
    st.subheader("Discriminative AI")
    st.write("Learns to classify data or predict labels.")
    st.info("Example: A spam classifier predicts spam or not spam.")

st.subheader("Key Differences")

comparison = pd.DataFrame({
    "Feature": ["Purpose", "Output", "Learning focus", "Example"],
    "Generative AI": [
        "Generate new data",
        "New content",
        "Data distribution",
        "GPT-2"
    ],
    "Discriminative AI": [
        "Classify data",
        "Labels or predictions",
        "Decision boundaries",
        "Logistic Regression"
    ]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)
st.divider()

st.header("8. Understanding Next-Token Probability")

st.write(
    "GPT-2 predicts the next token by calculating "
    "probabilities across its vocabulary."
)

st.subheader("How does it work?")

st.markdown("""
1. The user enters a prompt.
2. GPT-2 processes the input tokens.
3. The model produces logits for possible next tokens.
4. Softmax converts logits into probabilities.
5. The model can select a token based on the probability distribution.
""")

st.subheader("Important Concepts")

st.info(
    "**Logits:** Raw scores produced by the model before softmax."
)

st.info(
    "**Softmax:** Converts logits into a probability distribution "
    "whose values sum to 1."
)

st.info(
    "**Temperature:** Controls how sharp or flat the probability "
    "distribution becomes."
)

st.info(
    "**Top-K and Top-P:** Restrict the candidate tokens considered "
    "during sampling."
)

st.caption(
    "Note: GPT-2 predicts tokens, which may be complete words, "
    "word fragments or punctuation."
)
st.divider()

st.header("9. GPT-2 Self-Attention Visualization")

st.write(
    "Explore how GPT-2 distributes attention across "
    "different tokens in a sentence."
)

attention_prompt = st.text_input(
    "Enter a sentence for attention visualization:",
    value="The cat sat on the mat because it was tired.",
    key="attention_prompt"
)

col1, col2 = st.columns(2)

with col1:
    layer = st.selectbox(
        "Select Transformer Layer",
        options=list(range(12)),
        index=11,
        key="attention_layer"
    )

with col2:
    head = st.selectbox(
        "Select Attention Head",
        options=list(range(12)),
        index=0,
        key="attention_head"
    )

if st.button("Visualize Attention", type="primary"):

    if not attention_prompt.strip():
        st.warning("Please enter a sentence.")

    else:
        with st.spinner("Extracting attention weights..."):

            try:
                tokens, attention = get_attention_matrix(
                    attention_prompt,
                    layer=layer,
                    head=head
                )

                fig, ax = plt.subplots(figsize=(10, 8))

                heatmap = ax.imshow(
                    attention,
                    cmap="viridis",
                    aspect="auto"
                )

                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))

                ax.set_xticklabels(
                    tokens,
                    rotation=45,
                    ha="right"
                )

                ax.set_yticklabels(tokens)

                ax.set_xlabel("Tokens Being Attended To")
                ax.set_ylabel("Current Token")

                ax.set_title(
                    f"GPT-2 Attention Heatmap | "
                    f"Layer {layer} | Head {head}"
                )

                fig.colorbar(
                    heatmap,
                    ax=ax,
                    label="Attention Weight"
                )

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.success("Attention heatmap generated!")

                st.write("Tokens detected:", tokens)

            except Exception as e:
                st.error(f"Error generating attention: {e}")
                st.divider()

st.header("10. LSTM vs GPT-2 Perplexity Comparison")

st.write(
    "Compare how an LSTM and GPT-2 predict the next tokens "
    "in the same test sentence."
)

test_text = st.text_input(
    "Enter test sentence:",
    value=(
        "Neural networks can generate meaningful text "
        "by learning patterns from language data."
    ),
    key="perplexity_test_text"
)

if st.button("Compare Perplexity", key="compare_perplexity"):

    if not test_text.strip():
        st.warning("Please enter a test sentence.")

    else:
        with st.spinner("Calculating perplexity..."):

            device = torch.device("cpu")

            tokenizer, gpt2_model = load_model()

            input_ids = tokenizer(
                test_text,
                return_tensors="pt"
            )["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                st.warning("Please enter a longer sentence.")

            else:
                criterion = nn.CrossEntropyLoss()

                # GPT-2 evaluation
                gpt2_model.eval()

                with torch.no_grad():
                    gpt2_outputs = gpt2_model(input_ids)

                    gpt2_logits = gpt2_outputs.logits[:, :-1, :]
                    targets = input_ids[:, 1:]

                    gpt2_loss = criterion(
                        gpt2_logits.reshape(-1, tokenizer.vocab_size),
                        targets.reshape(-1)
                    )

                    gpt2_ppl = torch.exp(gpt2_loss).item()

                # LSTM evaluation
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

                    lstm_ppl = torch.exp(lstm_loss).item()

                # Results table
                results = pd.DataFrame({
                    "Model": ["GPT-2", "LSTM"],
                    "Perplexity": [gpt2_ppl, lstm_ppl]
                })

                st.subheader("Perplexity Results")

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                # Bar chart
                st.subheader("Visual Comparison")

                chart_data = results.set_index("Model")

                st.bar_chart(chart_data)

                # Interpretation
                st.subheader("Understanding the Results")

                if gpt2_ppl < lstm_ppl:
                    st.success(
                        f"GPT-2 achieved lower perplexity "
                        f"({gpt2_ppl:.2f}) than LSTM "
                        f"({lstm_ppl:.2f}) on this test sentence."
                    )
                elif lstm_ppl < gpt2_ppl:
                    st.success(
                        f"LSTM achieved lower perplexity "
                        f"({lstm_ppl:.2f}) than GPT-2 "
                        f"({gpt2_ppl:.2f}) on this test sentence."
                    )
                else:
                    st.info("Both models achieved equal perplexity.")

                st.caption(
                    "Note: GPT-2 is pretrained, whereas the LSTM "
                    "was trained on a small educational dataset. "
                    "This is not a controlled performance benchmark."
                )
                st.divider()

st.header("11. BERT vs GPT-2 Prediction Comparison")

st.write(
    "Compare BERT's masked-word predictions with "
    "GPT-2's next-token predictions."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("BERT - Masked Word Prediction")

    bert_sentence = st.text_input(
        "Enter a sentence with [MASK]:",
        value="The capital of France is [MASK].",
        key="bert_sentence"
    )

with col2:
    st.subheader("GPT-2 - Next Token Prediction")

    gpt_sentence = st.text_input(
        "Enter a prompt:",
        value="The capital of France is",
        key="gpt_sentence"
    )

if st.button("Compare BERT and GPT-2", key="compare_models"):

    try:
        with st.spinner("Generating predictions..."):

            bert_results = get_masked_word_predictions(
                bert_sentence,
                top_k=5
            )

            gpt_results = get_next_token_probabilities(
                gpt_sentence,
                top_k=5,
                temperature=1.0,
                top_p=1.0
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("BERT Predictions")

            for item in bert_results:
                st.write(
                    f"**{item['token']}** — "
                    f"{item['probability']:.2%}"
                )

        with col2:
            st.subheader("GPT-2 Predictions")

            for item in gpt_results:
                st.write(
                    f"**{item['token']}** — "
                    f"{item['probability']:.2%}"
                )

        st.success("Both models generated predictions!")

        st.info(
            "BERT predicts a masked token using surrounding context, "
            "while GPT-2 predicts the next token using preceding context. "
            "Their probabilities are not directly comparable."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        # ==========================================
# 12. U4 - PYTORCH TEXT SAMPLING
# ==========================================

st.header("12. PyTorch-Based Text Sampling")

st.write(
    "Explore how temperature and Top-K sampling "
    "change the text generated by GPT-2."
)

sampling_prompt = st.text_area(
    "Enter your starting prompt:",
    value="Artificial intelligence will",
    key="sampling_prompt"
)

col1, col2 = st.columns(2)

with col1:
    sampling_temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=1.5,
        value=1.0,
        step=0.1,
        key="sampling_temperature"
    )

with col2:
    sampling_top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=50,
        key="sampling_top_k"
    )

max_tokens = st.slider(
    "Maximum new tokens",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
    key="sampling_max_tokens"
)

if st.button("Generate Text", key="sampling_generate"):

    if not sampling_prompt.strip():
        st.warning("Please enter a prompt.")

    else:
        with st.spinner("GPT-2 is generating text..."):

            generated_text = generate_text(
                prompt=sampling_prompt,
                max_new_tokens=max_tokens,
                temperature=sampling_temperature,
                top_k=sampling_top_k
            )

        st.subheader("Generated Output")
        st.success(generated_text)

        st.caption(
            f"Temperature: {sampling_temperature} | "
            f"Top-K: {sampling_top_k} | "
            f"New token limit: {max_tokens}"
        )
        # ==========================================
# 13. U4 - GENERATED IMAGE OUTPUT
# ==========================================

st.header("13. AI-Generated Image Output")

st.write(
    "This image was generated by Stable Diffusion Turbo "
    "using text produced during our GPT-2 image-generation experiment."
)

import os

image_path = os.path.join(
    "outputs",
    "p5_generated_image.png"
)

if os.path.exists(image_path):
    st.image(
        image_path,
        caption="Image generated using GPT-2 text and Stable Diffusion Turbo",
        use_container_width=True
    )
else:
    st.warning("Generated image not found in the outputs folder.")
    # ==========================================
# 14. U5 - SAFETY FILTER DEMONSTRATION
# ==========================================

st.header("14. U5 - Safety Filter Demonstration")

st.write(
    "This section demonstrates how blocked words can be "
    "filtered during GPT-2 text generation."
)

prompt = st.text_input(
    "Enter your prompt:",
    value="In a small garden",
    key="safety_prompt"
)

blocked_input = st.text_input(
    "Words to block (comma-separated):",
    value="the",
    key="blocked_words"
)

if st.button("Generate Safe Text", key="safety_generate"):

    blocked_words = [
        word.strip()
        for word in blocked_input.split(",")
        if word.strip()
    ]

    with st.spinner("Generating filtered text..."):

        generated = generate_text(
            prompt=prompt,
            max_new_tokens=40,
            temperature=0.8,
            top_k=50,
            blocked_words=blocked_words
        )

    st.subheader("Generated Output")
    st.write(generated)

    st.success("Text generation with blocklist filtering completed!")

    st.caption(
        "Note: This is a basic token-blocking demonstration, "
        "not a complete toxicity detection or moderation system."
    )
    
# ==========================================
# 15. U5 - TEMPERATURE EVALUATION DASHBOARD
# ==========================================

st.header("15. U5 - Temperature Evaluation Dashboard")

st.write(
    "This experiment compares GPT-2 factual responses "
    "at low temperature (0.2) and high temperature (1.2)."
)

# Experiment results from manual evaluation
evaluation_data = {
    "Temperature": ["0.2", "1.2"],
    "Questions Tested": [10, 10],
    "Fully Correct": [0, 0],
    "Incorrect": [10, 10],
    "Factual Error Rate (%)": [100, 100]
}

import pandas as pd

evaluation_df = pd.DataFrame(evaluation_data)


st.subheader("Factual Evaluation Results")

st.table([
    {
        "Temperature": "0.2 (Low)",
        "Questions Tested": 10,
        "Fully Correct": 0,
        "Incorrect": 10,
        "Factual Error Rate": "100%"
    },
    {
        "Temperature": "1.2 (High)",
        "Questions Tested": 10,
        "Fully Correct": 0,
        "Incorrect": 10,
        "Factual Error Rate": "100%"
    }
])

st.subheader("Factual Error Rate Comparison")

chart_data = pd.DataFrame({
    "Temperature": ["0.2 (Low)", "1.2 (High)"],
    "Error Rate (%)": [100, 100]
})

st.bar_chart(
    chart_data,
    x="Temperature",
    y="Error Rate (%)",
    y_label="Factual Error Rate (%)"
)


st.subheader("Factual Error Rate Comparison")

st.bar_chart(
    evaluation_df.set_index("Temperature")[
        "Factual Error Rate (%)"
    ]
)

st.subheader("Observed Hallucination Examples")

failure_examples = pd.DataFrame({
    "Prompt": [
        "The chemical symbol for gold is",
        "The largest planet in our solar system is",
        "The capital of Japan is",
        "The author of Romeo and Juliet is"
    ],
    "GPT-2 Failure": [
        "Generated G instead of Au.",
        "Generated an incorrect planetary diameter.",
        "Generated information about the United States.",
        "Generated an unrelated author description."
    ]
})

st.table(failure_examples)

st.subheader("Experiment Interpretation")

st.write(
    "Both temperature settings produced factual errors "
    "in all 10 test cases under our strict manual evaluation."
)

st.info(
    "Temperature controls the randomness of token sampling. "
    "A lower temperature does not guarantee factual accuracy, "
    "and a higher temperature can produce less predictable outputs."
)

st.caption(
    "Limitation: Results are based on 10 prompts per temperature, "
    "one fixed sampling seed per prompt, and manual evaluation. "
    "They should not be treated as a general benchmark of GPT-2."
)
