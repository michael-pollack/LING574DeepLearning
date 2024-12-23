import numpy as np
import ops
import edugrad.nn as nn
from edugrad.tensor import Tensor


class DeepAveragingNetwork(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_labels: int
    ) -> None:
        super(DeepAveragingNetwork, self).__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_dim, bias=False)
        # 2 hidden layers
        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.output = nn.Linear(hidden_dim, num_labels)
        # hack for embedding initialization
        scale = 1 / np.sqrt(embedding_dim)
        self.embeddings.weight = Tensor(
            scale * (2 * np.random.random(self.embeddings.weight.value.shape) - 1)
        )

    def forward(self, bag_of_words: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass of DAN network.

        Args:
            bag_of_words: [batch_size, vocab_size] Tensor, with bag of words
                (including counts) for each example
            lengths: [batch_size] Tensor, with the length of each example stored

        Returns:
            [batch_size, num_classes] Tensor containing raw scores (logits) for
            each example in the batch
        """
        # TODO (~6 lines): implement here!
        # HINT: get averages, using lengths, and then pass through feedforward layers
        # ops.relu and ops.divide are your friends
        averages = ops.divide(ops.sum_along_columns(bag_of_words), lengths)
        hidden = ops.relu(self.hidden1(averages))
        hidden = ops.relu(self.hidden2(hidden))
        return self.output(hidden)