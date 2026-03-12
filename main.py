from datasets import load_dataset

from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocessing_normalisation import preprocess, normalise
from utils import (
    SEED,
    BATCH_SIZE,
    tokenizer,
    tokenize_data,
    calculate_metrics,
    plot_learning_curves,
)
from models import LSTMModel, CNNModel, train_model, get_predictions

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def main() -> None:

    # download the AG news dataset
    dataset_dict = load_dataset("SetFit/ag_news")
    """
    Dataset downloads the test and train split separately
    It save them in a dictionary where the first element
    is the train dataset, and the second one is the test dataset

    they are both Dataset objects containing:
        features: text label label_text
        num_rows: int
    """
    # Preprocessing
    train_full = dataset_dict["train"].to_pandas()
    test = dataset_dict["test"].to_pandas()
    train_full["text"] = train_full["text"].apply(preprocess).apply(normalise)
    test["text"] = test["text"].apply(preprocess).apply(normalise)

    print("Loaded data head (before split):")
    print(train_full.head(5))

    # split the training dataset into train and validation
    train, dev = train_test_split(
        train_full,
        test_size=0.16,
        random_state=SEED,
        stratify=train_full["label"]
    )

    print(f"Split sizes: Train({len(train)}), Dev({len(dev)}), Test({len(test)})")

    # Tokenization (Replacing TF-IDF/TorchText)
    # X_train, X_dev, and X_test are now dictionaries containing 'input_ids' and 'attention_mask'

    print("Tokenizing datasets...")
    X_train = tokenize_data(train["text"])
    X_dev = tokenize_data(dev["text"])
    X_test = tokenize_data(test["text"])

    # Y labels
    y_train = torch.tensor(train["label"].values)
    y_dev = torch.tensor(dev["label"].values)
    y_test = torch.tensor(test["label"].values)

    # Batching
    train_dataset = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
    dev_dataset = TensorDataset(X_dev['input_ids'], X_dev['attention_mask'], y_dev)
    test_dataset = TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    lstm = LSTMModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=4,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
    )

    cnn = CNNModel(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=100, 
        num_classes=4
    )
    
    
    trained_lstm, hist = train_model(lstm, train_loader, dev_loader)
    plot_learning_curves(hist, "LSTM")
    test_predictions = get_predictions(trained_lstm, test_loader)
    calculate_metrics(test, test_predictions, "LSTM")
    
    trained_cnn, cnn_hist = train_model(cnn, train_loader, dev_loader)
    plot_learning_curves(cnn_hist, "CNN")
    test_predictions_cnn = get_predictions(trained_cnn, test_loader)
    calculate_metrics(test, test_predictions_cnn, "CNN")


if __name__ == "__main__":
    main()
