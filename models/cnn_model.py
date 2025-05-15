import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM, n_filters=config.N_FILTERS_CNN,
                 filter_sizes=config.FILTER_SIZES_CNN, output_dim=config.OUTPUT_DIM,
                 dropout=config.DROPOUT_RATE, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Create multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        # Calculate the total number of features after convolution and pooling
        fc_in_dim = len(filter_sizes) * n_filters
        self.fc = nn.Linear(fc_in_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, emb dim]

        # Convolution expects [batch size, channels, seq len]
        # Channels = embedding dimension here
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, seq len]

        # Apply convolutions and pooling
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[n] = [batch size, n filters, seq len - filter_sizes[n] + 1]

        # Apply max pooling over time (the sequence length dimension)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[n] = [batch size, n filters]

        # Concatenate the pooled features from different filter sizes
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n filters * len(filter_sizes)]

        return self.fc(cat)

    



class DynamicMaxPool1d(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        # x: [batch size, channels, seq len]
        k = min(self.k, x.shape[2])  # Ensure k does not exceed sequence length
        out, _ = x.topk(k, dim=2)
        return out

class DCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM, 
                 n_filters=config.N_FILTERS_CNN, 
                 filter_sizes=config.FILTER_SIZES_CNN, 
                 output_dim=config.OUTPUT_DIM, 
                 dropout=config.DROPOUT_RATE, 
                 pad_idx=0, 
                 k_dynamic=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Use a single convolution layer for simplicity
        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=n_filters,
                              kernel_size=filter_sizes[0])  # Using the first filter size for main conv

        # Dynamic pooling layer
        self.dynamic_pool = DynamicMaxPool1d(k=k_dynamic)

        self.fc = nn.Linear(n_filters * k_dynamic, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch size, seq len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        # embedded: [batch size, emb dim, seq len]

        conved = F.relu(self.conv(embedded))
        # conved: [batch size, n_filters, seq_len - kernel_size + 1]

        pooled = self.dynamic_pool(conved)
        # pooled: [batch size, n_filters, k_dynamic]

        pooled_flat = pooled.view(pooled.size(0), -1)
        # pooled_flat: [batch size, n_filters * k_dynamic]

        output = self.fc(self.dropout(pooled_flat))
        return output


