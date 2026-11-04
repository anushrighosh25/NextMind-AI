from utils.bert_utils import get_masked_word_predictions


print("\nBERT Masked Word Prediction Test")
print("--------------------------------")

sentence = "The capital of France is [MASK]."

print("\nInput sentence:")
print(sentence)

predictions = get_masked_word_predictions(
    sentence,
    top_k=5
)

print("\nTop 5 BERT Predictions:")

for item in predictions:
    print(
        f"{item['token']} : "
        f"{item['probability']:.4f}"
    )