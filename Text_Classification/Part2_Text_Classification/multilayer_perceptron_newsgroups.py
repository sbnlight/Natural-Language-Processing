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
import copy  # Used for deep copying the best model weights
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple
import re  # Used for regex-based text cleaning
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
    
    # Load stopwords. Stopwords are common words (like "the", "is") 
    # that usually carry little meaning in classification.
    try:
        STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])
    except:
        STOP_WORDS = set()

    def _pre_process_text(self, text: str) -> List[str]:
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters (keep only a-z, 0-9) to reduce noise
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        
        # Filter out stopwords and very short words
        # This reduces the vocabulary size and focuses the model on meaningful content words
        filtered_words = [
            w for w in words 
            if w not in self.STOP_WORDS and len(w) > 2
        ]
        return filtered_words

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # offset because padding index is 0
        self.token2id = {t: (i + 1) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        # Use the enhanced _pre_process_text method defined above
        tokens = self._pre_process_text(text)
        ids = []
        for t in tokens:
            if t in self.token2id:
                ids.append(self.token2id[t])
            # Unknown words are implicitly ignored
        return ids


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels file and returns the mapping."""
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
        max_length: int = 200, # Use 200 because news articles are longer than tweets
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

        All of have type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        
        token_ids = self.tokenizer.tokenize(dp.text)
        
        # Truncate or Pad
        length = len(token_ids)
        if length > self.max_length:
            token_ids = token_ids[:self.max_length]
            length = self.max_length
        else:
            token_ids = token_ids + [Tokenizer.TOK_PADDING_INDEX] * (self.max_length - length)
            
        label_id = self.label2id[dp.label] if dp.label else -1
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(self, vocab_size: int, num_classes: int, padding_index: int):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.padding_index = padding_index
        
        # Architecture Hyperparameters
        embed_dim = 128
        hidden_dim1 = 512
        hidden_dim2 = 256
        
        # EmbeddingBag is efficient for NBOW. mode='mean' averages the word embeddings.
        self.embedding = nn.EmbeddingBag(
            vocab_size + 1, 
            embed_dim, 
            mode='mean', 
            padding_idx=padding_index
        )
        
        # Deep Network Architecture (2 Hidden Layers)
        # Structure: Input -> Embed -> FC1 -> ReLU -> Dropout -> FC2 -> ReLU -> Dropout -> FC3 -> Output
        self.fc1 = nn.Linear(embed_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        
        # use 0.5 to prevent overfitting on the small dataset
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Inputs:
            input_features_b_l (tensor): Input data for an example or a batch of examples.
            input_length (tensor): The length of the input data.

        Returns:
            output_b_c: The output of the model.
        """
        embedded = self.embedding(input_features_b_l)
        
        # Pass through Layer 1
        h = self.fc1(embedded)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Pass through Layer 2
        h = self.fc2(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Output Layer
        out = self.fc3(h)
        return out


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, data: BOWDataset) -> List[int]:
        """Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        self.model.eval()
        all_predictions = []
        # Batch size for prediction can be larger since we don't store gradients
        dataloader = DataLoader(data, batch_size=64, shuffle=False)
        
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
            The accuracy of the model.
        """
        self.model.eval()
        dataloader = DataLoader(data, batch_size=64, shuffle=False)
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
        """Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        # Added Label Smoothing to cross-entropy loss.
        # This prevents the model from becoming too confident, helping generalization.
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        torch.manual_seed(0)
        
        # [MODIFIED] Variables to track the best model state
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
            
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                outputs = self.model(inputs_b_l, lengths_b)
                loss = criterion(outputs, labels_b)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            per_dp_loss = total_loss / len(dataloader)

            # Evaluate on validation data
            self.model.eval()
            val_acc = self.evaluate(val_data)

            print(
                f"Epoch: {epoch + 1:<2} | Loss: {per_dp_loss:.4f} | Val accuracy: {100 * val_acc:.2f}%"
            )
            
            # Checkpointing logic:
            # If the current epoch's validation accuracy is better than the best we've seen,
            # save the model weights. This ensures we don't return an overfitted model from the last epoch.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Deep copy is essential here to store the actual values, not just a reference
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  >>> Best model updated: {100 * best_val_acc:.2f}%")

        # Restore the best weights found during training
        if best_model_state is not None:
            print(f"Restoring best model with Val Acc: {100 * best_val_acc:.2f}%")
            self.model.load_state_dict(best_model_state)


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
        "-e", "--epochs", type=int, default=20, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    # Increase vocab size to 25,000 to capture more rare but informative words
    tokenizer = Tokenizer(train_data, max_vocab_size=25000)
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    # Increase max_length to 200 for longer news articles
    max_length = 200
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
    # Add weight_decay=1e-5 (L2 Regularization).
    # It penalizes large weights, preventing the model from overfitting to noise in the training data.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    trainer.train(train_ds, val_ds, optimizer, num_epochs)

    # Evaluate on dev
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on test
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )