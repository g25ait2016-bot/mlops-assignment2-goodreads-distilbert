"""Simple inference script for the published DistilBERT Goodreads genre classifier.

Usage:
    python inference.py --text "This book had strong characters and a suspenseful plot."
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_ID = "Aukrk/distilbert-goodreads-genres"


def predict(text: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier(text, truncation=True, max_length=512)


def main():
    parser = argparse.ArgumentParser(description="Predict Goodreads review genre using a fine-tuned DistilBERT model.")
    parser.add_argument("--text", required=True, help="Review text to classify")
    args = parser.parse_args()

    result = predict(args.text)
    print(result)


if __name__ == "__main__":
    main()
