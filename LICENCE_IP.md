
# NextMind AI – Licence and Intellectual Property Note

## 1. Overview

NextMind AI is an educational Generative AI project that integrates pretrained models and open-source software libraries.

The project acknowledges the original developers and licence conditions of all third-party resources used.

## 2. Pretrained Model Licences

| Model | Developer | Licence / Terms |
|---|---|---|
| GPT-2 | OpenAI | MIT Licence |
| BERT Base Uncased | Google | Apache License 2.0 |
| Stable Diffusion Turbo | Stability AI | Stability AI Community License Agreement |
| Custom LSTM | Developed for this project | Project-specific implementation |

Official model references:

- GPT-2: https://huggingface.co/openai-community/gpt2
- BERT: https://huggingface.co/google-bert/bert-base-uncased
- Stable Diffusion Turbo: https://huggingface.co/stabilityai/sd-turbo

## 3. Software Libraries

| Library | Purpose | Licence |
|---|---|---|
| Python | Programming language | Python Software Foundation Licence |
| PyTorch | Deep learning framework | BSD-style licence |
| Hugging Face Transformers | Model integration | Apache License 2.0 |
| Streamlit | Web application | Apache License 2.0 |
| NumPy | Numerical computing | BSD-style licence |
| Pandas | Data processing | BSD-style licence |
| Matplotlib | Data visualization | Matplotlib Licence |

The exact licence and version of each dependency should be checked against its official distribution.

## 4. Dataset and Training Data

The custom LSTM model was trained on eight manually written educational sentences embedded in `utils/train_lstm.py`.

No external dataset was downloaded for this experiment.

The GPT-2 tokenizer was used to convert the training text into token IDs. The tokenizer's licence and terms remain applicable to the tokenizer itself.

GPT-2 and BERT use pretrained model weights and their original training data. The project does not claim ownership of those original datasets.

## 5. Intellectual Property and Attribution

- The project-specific application code and experiments were developed for NextMind AI.
- Third-party pretrained models remain the property of their respective owners and are subject to their licences.
- Original developers and model repositories are acknowledged.
- Model weights and third-party software must not be redistributed without following their applicable licence conditions.
- Generated text and images may have separate intellectual property considerations.

## 6. Ethical and Responsible Use

NextMind AI is intended for academic and educational purposes.

Users should:
- Verify AI-generated factual information.
- Avoid entering confidential or personal information.
- Respect copyright and intellectual property.
- Review generated images and text before publishing or sharing.
- Avoid using generated content for harmful or misleading purposes.

## 7. Disclaimer

This document is an academic summary of attribution and licensing considerations, not legal advice.

Users must consult the original model licences and terms before commercial use, redistribution or publication.
