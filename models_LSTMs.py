# Importing Main Libraries
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


# LSTM Model Fixed Input
class LSTM_fixed_len(torch.nn.Module):
    """
    Initializing LSTM Model for Fixed Input
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Embedding Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM Layer
        self.linear = nn.Linear(hidden_dim, output_size)    # Linear Layer
        self.dropout = nn.Dropout(0.2)  # Dropout

    # Forward Method to use while Training and Testing
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


# LSTM Model Variable Input
class LSTM_variable_input(torch.nn.Module):
    """
    Initializing LSTM Model for Variable Input in Padded Sequence Layer
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)  # Dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Embedding Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM Layer
        self.linear = nn.Linear(hidden_dim, output_size)    # Linear Layer

    # Forward Method to use while Training and Testing
    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = fix_pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


# LSTM Model Fixed Input with Stanford Glove Embeddings
class LSTM_glove_vecs(torch.nn.Module):
    """
    Initializing LSTM Model with glove weights in Embedding Layer
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights, output_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Embedding Layer
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights)) # Glove Vectors inserted into Embedding Layer
        self.embeddings.weight.requires_grad = False  # freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM Layer
        self.linear = nn.Linear(hidden_dim, output_size)    # Linear Layer
        self.dropout = nn.Dropout(0.2)  # Dropout

    # Forward Method to use while Training and Testing
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


# Method added for fixed Padding Sequence
def fix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()  # Converted from cuda to cpu
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices, None)
