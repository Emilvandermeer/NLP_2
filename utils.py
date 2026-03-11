from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    confusion_matrix,
    hinge_loss,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

SEED = 69
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
BATCH_SIZE = 64
MAX_LENGTH = 128
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(texts) -> AutoTokenizer:
    """
    Converts raw text into padded/truncated integer sequences.
    """
    #only using model for the tokenizer logic
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


def calculate_metrics(real: pd.DataFrame, pred: list,  model: str) -> None:
    print(f"\n{model} metrics:\n")
    print(f"Accuracy: {accuracy_score(real['label'], pred):.3f}")
    print(f"F1-Score: {f1_score(real['label'], pred, average='macro'):.3f}")
    print(f"Confusion Matrix: \n{confusion_matrix(real['label'], pred)}\n")

    cm = confusion_matrix(real['label'], pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=real['label_text'].unique()
    )
    disp.plot(xticks_rotation="vertical")
    plt.title(f"Confusion Matrix: {model}")
    plt.show()

    predictions = pd.DataFrame({
        "text": real["text"],
        "true_label": real["label"].map(LABELS),
        "pred_label": pd.Series(pred).map(LABELS),
    })

    errors = predictions[
        predictions["true_label"] != predictions["pred_label"]
    ]

    print(f"\nTotal Errors: {len(errors)}")
    print("Displaying first 20 misclassifications:\n")

    for i, doc in errors.head(20).iterrows():
        print(f"Article number {i}:")
        print(f"TRUE: {doc['true_label']} | PRED: {doc['pred_label']}")
        print(f"TEXT: {doc['text']}")
        print("-" * 80)

def plot_learning_curves(history: dict, model: str):


    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, history['dev_loss'], label='Dev Loss', marker='o', color='orange')
    
    plt.title(f'Learning Curves: {model}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()