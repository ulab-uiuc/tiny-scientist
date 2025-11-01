import argparse
import json
import os

import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def load_data():
    dataset = load_dataset("ag_news")
    train_data = dataset["train"].select(range(5000))
    val_data = dataset["train"].select(range(5000, 7000))
    test_data = dataset["test"].select(range(2000))

    vectorizer = TfidfVectorizer(max_features=512, stop_words="english")

    X_train = vectorizer.fit_transform(train_data["text"]).toarray()
    X_val = vectorizer.transform(val_data["text"]).toarray()
    X_test = vectorizer.transform(test_data["text"]).toarray()

    y_train = torch.tensor(train_data["label"], dtype=torch.long)
    y_val = torch.tensor(val_data["label"], dtype=torch.long)
    y_test = torch.tensor(test_data["label"], dtype=torch.long)

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
    def __init__(self, input_dim=512, hidden_units=128, output_dim=4):
        super(SingleLayerGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.gru(x.view(x.size(0), 1, -1))
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(5):
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

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()

    model = SingleLayerGRU(input_dim=512, hidden_units=128, output_dim=4)
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
