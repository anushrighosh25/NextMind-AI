from utils.gpt2_utils import get_attention_matrix

prompt = "The cat sat on the mat because it was tired."

tokens, attention = get_attention_matrix(
    prompt,
    layer=-1,
    head=0
)

print("\nGPT-2 Self-Attention Test")
print("------------------------")

print("\nTokens:")
print(tokens)

print("\nAttention Matrix Shape:")
print(attention.shape)

print("\nAttention Matrix:")
print(attention)

print("\nSum of attention weights for each token:")
print(attention.sum(axis=1))