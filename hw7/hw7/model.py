import torch
import torch.nn as nn

from torch import Tensor


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        padding_index: int,
        dropout: float = 0.0,
    ):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.padding_index = float(padding_index)
        self.dropout_prob = dropout
        if self.dropout_prob:
            self.dropout = nn.Dropout(dropout)

    def forward(self, characters: Tensor, lengths: list[int]):
        """Forward pass of an LSTM language model.

        Arguments:
            characters: [batch_size, max_seq_len] indices of characters
            lengths: batch_size length list of integer lengths for each example

        Returns:
            [batch_size, max_seq_len, vocab_size] Tensor
            output[i, j] is [vocab_size], the logits for next character prediction,
        """
        # [batch_size, max_seq_len, embedding_dim]
        embeddings = self.embedding(characters)
        if self.dropout_prob:
            embeddings = self.dropout(embeddings)
        # pack the sequences first; this prevents the RNN from computing on the padded tokens
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        # get packed sequence out of LSTM
        lstm_output, _ = self.lstm(packed_sequence)
        # unpack sequence
        # [batch_size, max_seq_len, hidden_dim]
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True, padding_value=self.padding_index
        )
        if self.dropout_prob:
            lstm_output = self.dropout(lstm_output)
        # final linear layer to get logits
        # [batch_size, max_seq_len, vocab_size]
        output = self.output(lstm_output)
        return output