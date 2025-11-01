import argparse
import json
import os

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def load_data():
    # Load the ag_news dataset
    dataset = load_dataset("ag_news")
    # Subsample the dataset
    train_data = dataset["train"].select(range(3500))
    val_data = dataset["train"].select(range(3500, 4500))
    test_data = dataset["test"].select(range(500))

    # Use Count Vectorizer for converting text data to vectors, ensuring a fixed input size of 500
    vectorizer = CountVectorizer(max_features=500, stop_words="english")

    X_train = vectorizer.fit_transform(train_data["text"]).toarray()
    X_val = vectorizer.transform(val_data["text"]).toarray()
    X_test = vectorizer.transform(test_data["text"]).toarray()

    y_train = torch.tensor(train_data["label"], dtype=torch.long)
    y_val = torch.tensor(val_data["label"], dtype=torch.long)
    y_test = torch.tensor(test_data["label"], dtype=torch.long)

    # Create DataLoaders for the datasets
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test),
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


class SingleLayerGRU(nn.Module):
    def __init__(
        self, input_dim=500, hidden_units=128, output_dim=4
    ):  # Increased hidden units
        super(SingleLayerGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(10):  # Increase epochs
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print(f"Training completed over {epoch+1} epochs.")


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute BLEU and ROUGE metrics
    bleu_score = corpus_bleu(
        [[str(l)] for l in all_labels], [str(p) for p in all_preds]
    )
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = [
        rouge_scorer_obj.score(str(l), str(p)) for l, p in zip(all_labels, all_preds)
    ]
    avg_rouge1 = sum([score["rouge1"].fmeasure for score in rouge_scores]) / len(
        rouge_scores
    )
    avg_rougeL = sum([score["rougeL"].fmeasure for score in rouge_scores]) / len(
        rouge_scores
    )

    return {"BLEU": bleu_score, "ROUGE-1": avg_rouge1, "ROUGE-L": avg_rougeL}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()

    model = SingleLayerGRU(
        input_dim=500, hidden_units=128, output_dim=4
    )  # Updated model
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, optimizer, criterion, device)
    metrics = evaluate(model, test_loader, device)

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print("Experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)
