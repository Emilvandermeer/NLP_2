from datasets import load_dataset
from preprocessing_normalisation import preprocess, normalise

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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
import torch
import torch.nn as nn

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") #only using model for the tokenizer logic
SEED = 69
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
BATCH_SIZE = 64
MAX_LENGTH = 128

def tokenize_data(texts):
    """
    Converts raw text into a batch of padded/truncated integer sequences.
    """
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


def evaluate_model(
        model: LogisticRegression | LinearSVC,
        X_train: list,
        X_dev: list,
        y_train: list,
        y_dev: list
        ) -> float:

    # train the model
    model.fit(X_train, y_train)

    if isinstance(model, LogisticRegression):
        train_pred = model.predict_proba(X_train)
        dev_pred = model.predict_proba(X_dev)
        # calculate the log loss
        train_loss = log_loss(y_train, train_pred)
        dev_loss = log_loss(y_dev, dev_pred)
    else:
        train_pred = model.decision_function(X_train)
        dev_pred = model.decision_function(X_dev)
        # calculate the squared hinge loss
        train_loss = hinge_loss(y_train, train_pred)**2
        dev_loss = hinge_loss(y_dev, dev_pred)**2

    print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
    return dev_loss


def tf_idf_tuning(
        train: pd.DataFrame,
        dev: pd.DataFrame,
        test: pd.DataFrame,
        model: LogisticRegression | LinearSVC,
        epochs: int = 15,
        ) -> tuple[list, list, list]:

    print("\nTuning max number of features...\n")

    max_features = 1000
    picked_tf_idf = None
    best = float('inf')

    for i in range(epochs):
        print(f"\nEpoch {i}:")

        tf_idf = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2)
        )

        # apply TF-IDF on the documents
        X_train = tf_idf.fit_transform(train["text"])
        X_dev = tf_idf.transform(dev["text"])

        dev_loss = evaluate_model(
                    model,
                    X_train,
                    X_dev,
                    train["label"],
                    dev["label"]
                    )

        # save the TF-IDF that leads to the smallest loss
        if dev_loss < best:
            best, picked_tf_idf = dev_loss, tf_idf

        max_features += 2000

    # apply best TF-IDF on the documents
    X_train = picked_tf_idf.fit_transform(train["text"])
    # do not fit on the dev and on test as it results in leakage
    X_dev = picked_tf_idf.transform(dev["text"])
    X_test = picked_tf_idf.transform(test["text"])

    return X_train, X_dev, X_test


def regularization_tuning(
        X_train: list,
        X_dev: list,
        y_train: list,
        y_dev: list,
        model: LogisticRegression | LinearSVC,
        epochs: int = 15,
        ) -> LogisticRegression | LinearSVC:

    print("\nTuning regularization factor...\n")
    c = 0.5
    picked_c = None
    best = float('inf')

    for i in range(epochs):
        print(f"\nEpoch {i}:")

        if isinstance(model, LogisticRegression):
            model = LogisticRegression(random_state=SEED, C=c, solver="saga")
        else:
            model = LinearSVC(loss="squared_hinge", C=c, random_state=SEED)

        dev_loss = evaluate_model(
                    model,
                    X_train,
                    X_dev,
                    y_train,
                    y_dev
                    )

        # save the C that leads to the smallest loss
        if dev_loss < best:
            best, picked_c = dev_loss, c

        c += 0.5

    # restore best C
    if isinstance(model, LogisticRegression):
        model = LogisticRegression(
            random_state=SEED,
            C=picked_c,
            solver="saga"
            )
    else:
        model = LinearSVC(loss="squared_hinge", C=picked_c, random_state=SEED)

    model.fit(X_train, y_train)
    return model


def run_model(
        model: LinearSVC | LogisticRegression,
        train: pd.DataFrame,
        dev: pd.DataFrame,
        test: pd.DataFrame,
        msg: str
        ) -> None:

    X_train, X_dev, X_test = tf_idf_tuning(
        train=train,
        dev=dev,
        test=test,
        model=model
    )

    model = regularization_tuning(
        X_train,
        X_dev,
        train["label"],
        dev["label"],
        model
    )

    lr_pred = model.predict(X_test)
    lr_dev = model.predict(X_dev)
    calculate_metrics(test, lr_pred, msg)
    calculate_metrics(dev, lr_dev, msg + " dev")


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

    # 4. Tokenization (Replacing TF-IDF/TorchText)
    # X_train, X_dev, and X_test are now dictionaries containing 'input_ids' and 'attention_mask'
    print("Tokenizing datasets...")
    X_train = tokenize_data(train["text"])
    X_dev = tokenize_data(dev["text"])
    X_test = tokenize_data(test["text"])

    # Y labels
    y_train = torch.tensor(train["label"].values)
    y_dev = torch.tensor(dev["label"].values)
    y_test = torch.tensor(test["label"].values)

    # Note: For Deep Learning, we usually wrap these in a TensorDataset + DataLoader 
    # so we can train in batches.
    print(f"Tokenization complete. Sequence shape: {X_train['input_ids'].shape}")

    print("\n--- Tokenization Sanity Check ---")
    sample_idx = 0
    print(f"Original Text: {train['text'].iloc[sample_idx][:100]}...")
    
    # Show how the tokenizer 'sees' it
    ids = X_train['input_ids'][sample_idx]
    tokens = tokenizer.convert_ids_to_tokens(ids[:20]) # First 20 tokens
    
    print(f"Tokens: {tokens}")
    print(f"Sequence Length: {len(ids)}")
    print("---------------------------------\n")


    #! MODELS HERE 


    #print("Split:")
    #print(f"\nTrain size: {len(train)} rows")
    #print(f"Dev size:   {len(dev)} rows")
    #print(f"Test size:  {len(test)} rows")

    #lr = LogisticRegression(random_state=SEED, solver="saga")
    #svm = LinearSVC(loss="squared_hinge", random_state=SEED)

    #run_model(lr, train, dev, test, "Logistic Regression")
    #run_model(svm, train, dev, test, "SVM")


if __name__ == "__main__":
    main()
