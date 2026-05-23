# MLOps Assignment 2 - Goodreads Genre Classification using DistilBERT

This repository contains the implementation for MLOps Assignment 2. The project fine-tunes a pre-trained DistilBERT model on Goodreads review data to classify book genres. The model training was performed in a Kaggle Notebook using GPU acceleration. Experiment tracking was done using Weights & Biases, and the final trained model was published to Hugging Face Hub.

## Project Workflow

The assignment follows an end-to-end MLOps workflow:

1. Import the starter notebook into Kaggle.
2. Enable GPU and Internet in Kaggle.
3. Store API credentials securely using Kaggle Secrets.
4. Load and sample Goodreads review data.
5. Encode text and labels for DistilBERT.
6. Fine-tune DistilBERT using Hugging Face Trainer.
7. Track training and evaluation metrics using Weights & Biases.
8. Save evaluation results as a W&B artifact.
9. Push the trained model and tokenizer to Hugging Face Hub.

## Model Used

The model used for this assignment is:

```text
distilbert-base-cased
```

DistilBERT was selected because it is smaller and faster than the full BERT model while still providing strong performance for text classification tasks. This makes it suitable for Kaggle GPU training and for demonstrating the MLOps workflow.

## Training Platform

The model was trained on Kaggle using GPU acceleration.

Kaggle Notebook:  
https://www.kaggle.com/code/anukumarkg25ait2016/anukumar-mlops-assn2-final

## Results

| Metric | Score |
|---|---:|
| Accuracy | 0.56937 |
| F1 Score | 0.57111 |
| Eval Loss | 2.4409 |

## Links

- Kaggle Notebook: https://www.kaggle.com/code/anukumarkg25ait2016/anukumar-mlops-assn2-final
- Hugging Face Model: https://huggingface.co/Aukrk/distilbert-goodreads-genres
- W&B Dashboard: https://wandb.ai/anuaussie-prom-iit-rajasthan/mlops-assignment2
- W&B Run: https://wandb.ai/anuaussie-prom-iit-rajasthan/mlops-assignment2/runs/l88vnpqs

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/g25ait2016-bot/mlops-assignment2-goodreads-distilbert.git
cd mlops-assignment2-goodreads-distilbert
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference using the published Hugging Face model:

```bash
python inference.py --text "This book had strong characters and a suspenseful plot."
```

The full training workflow is available in the public Kaggle notebook linked above. API tokens are not stored in this repository. W&B and Hugging Face tokens were managed securely using Kaggle Secrets.

## Repository Contents

```text
README.md
requirements.txt
inference.py
submission_links.txt
final_report.docx
final_report.pdf
```

## Notes

This assignment focuses on understanding the MLOps workflow rather than achieving maximum model accuracy. The key objective is to demonstrate model fine-tuning, experiment tracking, evaluation artifact logging, and model deployment to Hugging Face Hub.
