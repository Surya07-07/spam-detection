from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
model = AutoModelForSequenceClassification.from_pretrained("AntiSpamInstitute/spam-detector-bert-MoE-v2.2")

# Sample text
texts = [
    "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now.",
    "Hey, are we still meeting for lunch today?"
]

# Tokenize the input
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=1)

# Get predicted labels
predictions = torch.argmax(probabilities, dim=1)

# Map labels to class names
label_map = {0: "Not Spam", 1: "Spam"}
for text, prediction in zip(texts, predictions):
    print(f"Text: {text}\nPrediction: {label_map[prediction.item()]}\n")
