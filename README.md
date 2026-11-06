
# P5 – Next-Token Probability Lab

## Project Overview

The Next-Token Probability Lab demonstrates how GPT-2 predicts the next token and how sampling parameters influence text generation.

It also explores Transformer self-attention, compares GPT-2 with LSTM and BERT models, demonstrates text sampling and image generation, and evaluates factual errors and basic safety filtering.

## Project Objectives

- Visualize GPT-2 next-token probabilities.
- Understand temperature, Top-K and Top-P sampling.
- Visualize Transformer self-attention.
- Compare LSTM and GPT-2 using perplexity.
- Compare BERT masked-token predictions with GPT-2 next-token predictions.
- Implement PyTorch-based text sampling.
- Demonstrate AI image generation.
- Evaluate factual errors at different temperatures.
- Demonstrate basic token-blocking safety filters.

## Features

### U1 – Next-Token Probability
GPT-2 probability visualization with interactive temperature, Top-K and Top-P controls.

### U2 – Attention and Perplexity
Self-attention heatmaps and LSTM vs GPT-2 perplexity comparison.

### U3 – BERT vs GPT-2
Side-by-side masked-token and next-token predictions.

### U4 – Text and Image Generation
PyTorch text sampling and Stable Diffusion Turbo image generation.

### U5 – Responsible AI
Factual error evaluation at low and high temperatures and basic toxic-token filtering.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- GPT-2
- BERT
- LSTM
- Streamlit
- Matplotlib
- Pandas
- Stable Diffusion Turbo

## Project Structure

```text
P5_Next_Token_Probability_Lab/
├── app.py
├── utils/
├── pages/
├── outputs/
├── models/
├── requirements.txt
└── README.md
```

## Evaluation

The project includes:
- GPT-2 vs LSTM perplexity comparison.
- GPT-2 factual response evaluation at temperatures 0.2 and 1.2.
- Hallucination examples.
- Basic blocked-token filtering demonstration.

## Limitations

- The LSTM and GPT-2 perplexity comparison uses models trained on different amounts of data.
- The factual evaluation uses a small set of manually reviewed prompts.
- The safety filter is a basic blocklist, not a complete moderation system.
- The generated image is currently saved and displayed in the Streamlit app rather than generated dynamically within it.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Author

Anushri Ghosh

## Responsible AI Documentation

This project includes documentation covering model transparency, limitations, safety and intellectual property.

- [Model Card](MODEL_CARD.md)
- [Licence and Intellectual Property Note](LICENCE_IP.md)

These documents describe the pretrained models, experimental results, training data, licensing considerations and responsible use of NextMind AI.
