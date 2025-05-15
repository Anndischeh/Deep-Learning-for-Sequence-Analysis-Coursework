import torch
import torch.nn as nn
import config

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_RNN,
                 output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_RNN, dropout=config.DROPOUT_RATE,
                 pad_idx=0): # Assuming 0 is the padding index
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0) # Dropout only between layers
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, emb dim]
        output, hidden = self.rnn(embedded)
        # output = [batch size, seq len, hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]

        # We use the hidden state of the last layer
        hidden_last_layer = hidden[-1, :, :] # [batch size, hid dim]
        # Apply dropout before the final layer
        out = self.dropout(hidden_last_layer)
        return self.fc(out)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_LSTM,
                 output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_LSTM, bidirectional=config.BIDIRECTIONAL_LSTM,
                 dropout=config.DROPOUT_RATE, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, emb dim]

        # packed_output, (hidden, cell) = self.lstm(embedded) # Use if packing sequences
        # For padded sequences (simpler):
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = [batch size, seq len, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # cell = [n layers * num directions, batch size, hid dim]

        # Concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            # hidden is stacked [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer_1, ...]
            # We want the final hidden state from the last layer's forward and backward pass
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
            # hidden = [batch size, hid dim * 2]
        else:
            hidden = self.dropout(hidden[-1,:,:])
            # hidden = [batch size, hid dim]

        return self.fc(hidden)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM_GRU,
                 output_dim=config.OUTPUT_DIM, n_layers=config.N_LAYERS_GRU, bidirectional=config.BIDIRECTIONAL_LSTM, # Reuse BiLSTM setting
                 dropout=config.DROPOUT_RATE, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, emb dim]

        outputs, hidden = self.gru(embedded)
        # outputs = [batch size, seq len, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # Concatenate the final forward and backward hidden states
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        return self.fc(hidden)
