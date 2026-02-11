"""Multi-layer perceptron model for Assignment 1: Starter code.
You can change this code while keeping the function headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.
We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.
Dimension key:
b: batch size
l: max sequence length
c: number of classes
v: vocabulary size
For example,
feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
length_1 means a tensor of shape (1) == (1,).
loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""
import argparse
import os
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        # Convert text to lowercase and remove punctuation
        # Keep only alphanumeric characters and spaces to improve NBOW representation quality
        text = text.lower()
        # Replace non-alphanumeric characters with spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # Offset by 1 because padding index is 0
        self.token2id = {t: (i + 1) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        # Convert text to list of token IDs using the vocabulary mapping
        tokens = self._pre_process_text(text)
        ids = []
        for t in tokens:
            if t in self.token2id:
                ids.append(self.token2id[t])
            # Unknown words are simply ignored (not mapped to a special <UNK> token)
            # This is a design choice for simplicity; in production systems, <UNK> tokens are typically used
        return ids


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels from data and returns bidirectional mappings between labels and IDs."""
    labels = list(set([d.label for d in data]))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a single example as a tuple of torch.Tensors.
        features_l: The tokenized text of example, shaped (max_length,)
        length: The length of the text, shaped ()
        label: The label of the example, shaped ()

        All have type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        
        # Tokenize text to get token IDs
        token_ids = self.tokenizer.tokenize(dp.text)
        
        # Handle sequence length: truncate if too long, pad if too short
        length = len(token_ids)
        if length > self.max_length:
            token_ids = token_ids[:self.max_length]
            length = self.max_length
        else:
            # Pad with padding index to reach max_length
            token_ids = token_ids + [Tokenizer.TOK_PADDING_INDEX] * (self.max_length - length)
            
        # Convert label to ID; use -1 for unlabeled examples (e.g., test set)
        label_id = self.label2id[dp.label] if dp.label else -1
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),  # Length information (may be used for masking in advanced models)
            torch.tensor(label_id, dtype=torch.long)
        )


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification using Bag-of-Words representation."""

    def __init__(self, vocab_size: int, num_classes: int, padding_index: int):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
            padding_index (int): Index used for padding tokens (ignored in EmbeddingBag).
        """
        super().__init__()
        self.padding_index = padding_index
        
        embed_dim = 128
        hidden_dim = 256
        
        # Use EmbeddingBag for efficient bag-of-words representation
        # mode='mean' computes the average of embeddings for all tokens in the sequence
        # This effectively creates a fixed-size vector representation regardless of input length
        self.embedding = nn.EmbeddingBag(
            vocab_size + 1,  # +1 to account for padding index
            embed_dim,
            mode='mean',
            padding_idx=padding_index
        )
        
        # Two-layer feedforward network with dropout for regularization
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()  # ReLU activation provides non-linearity and avoids vanishing gradients
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Inputs:
            input_features_b_l (tensor): Input token IDs for a batch, shape (batch_size, sequence_length)
            input_length_b (tensor): Length of each sequence in the batch (unused here but kept for interface compatibility)

        Returns:
            output_b_c: Logits for each class, shape (batch_size, num_classes)
        """
        # EmbeddingBag automatically handles variable-length sequences and computes mean embedding
        embedded = self.embedding(input_features_b_l)
        
        # First hidden layer: linear transformation -> activation -> dropout
        h = self.fc1(embedded)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Output layer: project to class logits
        out = self.fc2(h)
        return out


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, data: BOWDataset) -> List[int]:
        """Predicts labels for all examples in the dataset.

        Inputs:
            data: Dataset containing examples to predict.

        Returns:
            List of predicted class IDs.
        """
        self.model.eval()
        all_predictions = []
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for inputs, lengths, _ in dataloader:
                outputs = self.model(inputs, lengths)
                preds = torch.argmax(outputs, dim=1)
                all_predictions.extend(preds.tolist())
                
        return all_predictions

    def evaluate(self, data: BOWDataset) -> float:
        """Evaluates the model on a dataset.

        Inputs:
            data: The dataset to evaluate on.

        Returns:
            The accuracy of the model (float between 0 and 1).
        """
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, lengths, labels in dataloader:
                outputs = self.model(inputs, lengths)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return correct / total if total > 0 else 0.0

    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> None:
        """Trains the MLP model using cross-entropy loss.

        Inputs:
            training_data: Training dataset.
            val_data: Validation dataset for monitoring progress.
            optimizer: Optimization algorithm (e.g., Adam).
            num_epochs: Number of training epochs.
        """
        criterion = nn.CrossEntropyLoss()
        
        # Set random seed for reproducibility
        torch.manual_seed(0)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader):
                optimizer.zero_grad()
                outputs = self.model(inputs_b_l, lengths_b)
                loss = criterion(outputs, labels_b)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            per_dp_loss = total_loss / len(dataloader)

            # Evaluate on validation set after each epoch
            self.model.eval()
            val_acc = self.evaluate(val_data)

            print(
                f"Epoch: {epoch + 1: <2} | Loss: {per_dp_loss:.2f} | Val accuracy: {100 * val_acc:.2f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="newsgroups",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()
    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    # Build vocabulary from training data only (to prevent data leakage)
    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
    )

    trainer = Trainer(model)

    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer.train(train_ds, val_ds, optimizer, num_epochs)

    # Evaluate on development set
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Generate predictions for test set and save to CSV
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )