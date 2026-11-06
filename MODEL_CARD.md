
# NextMind AI – Model Card

## 1. Project Overview

**Project Name:** NextMind AI – Next-Token Probability Lab

**Project Type:** Educational Generative AI Application

**Framework:** Python, Streamlit, PyTorch and Hugging Face Transformers

NextMind AI is an educational application that demonstrates how Generative AI models predict text, calculate token probabilities, use attention mechanisms, and generate text and images.

The application also explores model limitations, hallucinations, and basic AI safety techniques.

## 2. Intended Use

NextMind AI is designed for:

- Learning how Generative AI works.
- Understanding next-token prediction and probability distributions.
- Comparing GPT-style and BERT-family models.
- Visualizing Transformer attention.
- Demonstrating text and image generation.
- Exploring hallucinations and basic AI safety.

The application is intended for educational and experimental purposes only. It is not designed for medical, legal, financial, or other high-stakes decisions.


## 3. Models Used

NextMind AI integrates four models for different Generative AI tasks.

| Model | Model Identifier | Purpose |
|---|---|---|
| GPT-2 | openai-community/gpt2 | Next-token prediction and text generation |
| BERT Base Uncased | google-bert/bert-base-uncased | Masked-token prediction |
| Stable Diffusion Turbo | stabilityai/sd-turbo | Text-to-image generation |
| Custom LSTM | Project-trained model | Experimental language modelling and perplexity comparison |

### 3.1 GPT-2

- Architecture: Transformer decoder.
- Task: Autoregressive language generation.
- Project usage: Predicts the next token based on preceding text and generates text.
- Limitation: May produce incorrect, irrelevant or fabricated information.

### 3.2 BERT Base Uncased

- Architecture: Bidirectional Transformer encoder.
- Task: Masked language modelling.
- Project usage: Predicts masked words using surrounding context.
- Limitation: Designed for masked-token prediction, not ordinary left-to-right text generation.

### 3.3 Stable Diffusion Turbo

- Architecture: Diffusion-based image generation model.
- Task: Text-to-image generation.
- Project usage: Generates images from text prompts.
- Limitation: Generated images may contain visual inaccuracies, bias or unintended content.

### 3.4 Custom LSTM

- Architecture: Long Short-Term Memory neural network.
- Task: Experimental language modelling.
- Project usage: Trained for five epochs on a small text corpus and compared with GPT-2 using perplexity.
- Limitation: Limited training data and training duration affect its ability to generalize.

## 4. Evaluation and Experimental Results

### 4.1 GPT-2 vs Custom LSTM Perplexity

The project compares GPT-2 and a custom-trained LSTM using perplexity.

| Model | Observed Perplexity |
|---|---:|
| GPT-2 | 169.44 |
| Custom LSTM | 2467.75 |

A lower perplexity generally indicates better prediction of the evaluated text.

**Observation:** GPT-2 achieved lower perplexity than the custom LSTM in this experiment.

**Limitation:** This is a small experimental comparison, not a controlled benchmark. GPT-2 was pretrained on a large corpus, whereas the LSTM was trained for only five epochs on a small corpus.

### 4.2 Hallucination Experiment

GPT-2 was tested using 10 prompts at two temperature settings.

| Temperature | Observed Error Rate |
|---|---:|
| 0.2 | 100% |
| 1.2 | 100% |

The experiment used strict all-or-nothing manual scoring.

Some observed failure examples included:

- Incorrect chemical symbol for gold.
- Incorrect planetary diameter.
- An unrelated answer to a question about Japan's capital.
- Incorrect information about the author of Romeo and Juliet.

**Observation:** GPT-2 produced factual errors at both temperature settings.

**Limitation:** This experiment uses only 10 prompts per temperature and does not represent the general factual accuracy of GPT-2.

## 5. Safety and Responsible AI

The application includes a basic token blocklist filtering mechanism.

Its purpose is to prevent selected blocked words or token sequences from being generated.

However, the mechanism has limitations:

- It is not a complete toxicity detection system.
- It may miss spelling variations and subword combinations.
- It does not guarantee safe or unbiased output.
- Generated text should be reviewed before reuse.

The hallucination experiment demonstrates why AI-generated information should be independently verified.

## 6. Licence, Intellectual Property and Attribution

NextMind AI uses third-party pretrained models and open-source software. The original developers retain ownership of their respective models, software and training materials.

### 6.1 Pretrained Model Licences

| Model | Licence / Usage Conditions |
|---|---|
| GPT-2 | MIT Licence |
| BERT Base Uncased | Apache License 2.0 |
| Stable Diffusion Turbo | Refer to the Stability AI licence and model terms |
| Custom LSTM | Project-specific implementation; training data rights must be verified |

Official model references:

- GPT-2: https://huggingface.co/openai-community/gpt2
- BERT: https://huggingface.co/google-bert/bert-base-uncased
- SD-Turbo: https://huggingface.co/stabilityai/sd-turbo

### 6.2 Intellectual Property

- The project-specific application code and experiments were developed for NextMind AI.
- The project does not claim ownership of GPT-2, BERT or Stable Diffusion Turbo.
- Original model developers and repositories are acknowledged.
- Third-party model weights and software remain subject to their respective licence conditions.
- Generated text and images should be reviewed before redistribution or publication.


### 6.3 Dataset and Training Data

The custom LSTM model was trained using a small, manually written educational text corpus embedded directly in the project's training script (`utils/train_lstm.py`).

The corpus contains eight sentences covering:
- Artificial intelligence
- Machine learning
- Deep learning
- Natural language processing
- Language models
- Transformer architectures
- LSTM networks
- Python

No external training dataset was downloaded for this LSTM experiment.

The text was tokenized using the pretrained GPT-2 tokenizer. The model was trained for five epochs using a sequence length of 16 and a batch size of 4.

The training corpus was created specifically for this educational experiment. It is not a representative language dataset and is insufficient for training a general-purpose language model.

GPT-2 and BERT use their original pretrained weights and training data. The original training datasets are not owned or redistributed by this project.

### 6.4 Ethical Use

NextMind AI is intended for educational and experimental purposes.

Users should:
- Verify AI-generated factual information.
- Avoid entering confidential or personal information.
- Respect copyright and intellectual property.
- Review generated content before sharing it.
- Avoid using AI-generated content for harmful or misleading purposes.

### 6.5 Disclaimer

This document provides an academic summary of model attribution and licensing considerations. It is not legal advice. Users should consult the original licence documents and terms before commercial use or redistribution.
