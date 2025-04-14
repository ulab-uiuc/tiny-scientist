import argparse
import os
from typing import Tuple

import torch
import torch.optim as optim
from datasets import Dataset, load_dataset
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

# Define model and dataset
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "glue"
TASK_NAME = "sst2"


def load_data() -> Tuple[Dataset, Dataset]:
    """Loads the dataset and prepares train/test splits."""
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples: Dataset) -> Dataset:
        return tokenizer(examples["sentence"], truncation=True, padding="max_length")

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset["train"], dataset["validation"]


class AdaptiveLRModel(Module):  # type: ignore[misc]
    """Custom model wrapper for adaptive learning rate experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BaseModelOutput:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def train_and_evaluate(
    output_dir: str, initial_lr: float = 5e-5, adapt_lr: bool = True
) -> None:
    """Trains the model with adaptive learning rates and evaluates performance."""
    train_data, val_data = load_data()
    model = AdaptiveLRModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        if adapt_lr
        else None
    )

    loss_fn = CrossEntropyLoss()
    train_loader: DataLoader[Dataset] = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader: DataLoader[Dataset] = DataLoader(val_data, batch_size=16)

    best_val_loss = float("inf")

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            loss = loss_fn(outputs.logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                loss = loss_fn(outputs.logits, batch["label"].to(device))
                val_loss += loss.item()

        if scheduler:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1} - Training Loss: {running_loss / len(train_loader):.4f}, "
            f"Validation Loss: {val_loss / len(val_loader):.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment with Adaptive Learning Rate"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for model checkpoints",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument(
        "--adaptive", action="store_true", help="Use adaptive learning rates"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_and_evaluate(args.out_dir, initial_lr=args.lr, adapt_lr=args.adaptive)
