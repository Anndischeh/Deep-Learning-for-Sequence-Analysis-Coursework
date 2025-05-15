import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import config

class IMDBDataset(Dataset):
    """Custom PyTorch Dataset for IMDB sequences."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Ensure sequence is a tensor
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        # Ensure label is a tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float) # Use float for BCEWithLogitsLoss
        return sequence, label

class IMDBDatasetBert(Dataset):
    """Custom PyTorch Dataset for BERT-based models."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract tensors for the specific item
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Ensure label is a tensor
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float) # Use float for BCEWithLogitsLoss
        return item # Return dict expected by Hugging Face models



def create_dataloaders(data, labels, batch_size, test_size=config.TEST_SIZE, val_size=config.VALIDATION_SIZE, is_bert=False, tokenizer=None):
    """Splits data and creates PyTorch DataLoaders."""

    print(f"Splitting data: Test size={test_size}, Validation size={val_size} (of train set)")

    # Ensure labels are numeric (0 or 1)
    labels_numeric = np.array([1 if label == config.POSITIVE_LABEL else 0 for label in labels])

    # Split into Train+Validation and Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels_numeric, test_size=test_size, random_state=42, stratify=labels_numeric
    )

    # Split Train+Validation into Train and Validation
    # Adjust val_size relative to the size of the train_val set
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=42, stratify=y_train_val
    )

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    if is_bert:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for BERT dataloaders.")

        print("Tokenizing data for BERT...")
        # Tokenize using the provided Hugging Face tokenizer
        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding='max_length', max_length=config.MAX_LEN, return_tensors="pt")
        val_encodings = tokenizer(X_val.tolist(), truncation=True, padding='max_length', max_length=config.MAX_LEN, return_tensors="pt")
        test_encodings = tokenizer(X_test.tolist(), truncation=True, padding='max_length', max_length=config.MAX_LEN, return_tensors="pt")

        # Create Datasets
        train_dataset = IMDBDatasetBert(train_encodings, y_train)
        val_dataset = IMDBDatasetBert(val_encodings, y_val)
        test_dataset = IMDBDatasetBert(test_encodings, y_test)

        # DataLoaders don't need collate_fn if data is already tensors
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else: # For RNN, CNN models using pre-computed sequences
        # Pad sequences (assuming X_train, X_val, X_test contain lists of indices)
        # This padding happens *before* creating the Dataset
        print("Padding sequences...")
        X_train_pad = pad_sequences_custom(X_train, maxlen=config.MAX_LEN, padding=config.PAD_TYPE, truncating=config.TRUNC_TYPE)
        X_val_pad = pad_sequences_custom(X_val, maxlen=config.MAX_LEN, padding=config.PAD_TYPE, truncating=config.TRUNC_TYPE)
        X_test_pad = pad_sequences_custom(X_test, maxlen=config.MAX_LEN, padding=config.PAD_TYPE, truncating=config.TRUNC_TYPE)


        # Create Datasets
        train_dataset = IMDBDataset(X_train_pad, y_train)
        val_dataset = IMDBDataset(X_val_pad, y_val)
        test_dataset = IMDBDataset(X_test_pad, y_test)

        # Define collate_fn for padding within the batch (Alternative to pre-padding)
        # def collate_batch(batch):
        #     label_list, text_list = [], []
        #     for (_text, _label) in batch:
        #         label_list.append(_label)
        #         # Process text if needed, assume it's already tensor of indices
        #         text_list.append(_text)
        #     # Pad sequences within the batch
        #     text_list_padded = pad_sequence(text_list, batch_first=True, padding_value=0) # Assuming 0 is pad index
        #     label_list = torch.tensor(label_list, dtype=torch.float)
        #     return text_list_padded, label_list

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Removed collate_fn as we pre-padd
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader


def pad_sequences_custom(sequences, maxlen, padding='post', truncating='post', value=0):
    """Pads/truncates a list of sequences (lists of numbers)."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                truncated_seq = seq[-maxlen:]
            else: # 'post'
                truncated_seq = seq[:maxlen]
            padded_sequences.append(truncated_seq)
        else:
            padding_len = maxlen - len(seq)
            padding_values = [value] * padding_len
            if padding == 'pre':
                padded_seq = padding_values + seq
            else: # 'post'
                padded_seq = seq + padding_values
            padded_sequences.append(padded_seq)
    return padded_sequences

