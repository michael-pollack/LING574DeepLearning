import random
import time

import data
import ops
import util
import edugrad.nn as nn
import numpy as np
from edugrad.ops import reduce_mean
from edugrad.optim import SGD
from edugrad.tensor import Tensor
from vocabulary import Vocabulary


def bce_loss(probabilities: Tensor, labels: Tensor) -> Tensor:
    """Compute binary cross entropy.

    Hint: use ops.multiply, ops.log, + and - on tensors, as well as
    reduce_mean (imported from edugrad.ops above).

    Args:
        probabilities: [batch_size] in [0, 1]
        labels: [batch_size] in {0, 1}

    Returns:
        mean of
        -(y * log(yhat) + (1-y) * log(1 - yhat))
        where yhat is the probabilities and y the labels
    """
    # helper Tensors
    shape = probabilities.value.shape
    ones = Tensor(np.ones(shape))
    minus_one = Tensor(np.array(-1.0))
    # TODO: implement here
    # We recommend breaking down in the following way:
    # compute each side of the `+`, then add, then take mean, then get
    # the negative value
    pos_side = ops.multiply(labels, ops.log(probabilities))
    neg_side = ops.multiply((ones - labels), ops.log(ones - probabilities))
    total_sum = pos_side + neg_side
    mean = reduce_mean(total_sum)
    return ops.multiply(mean, minus_one)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Embedding, self).__init__()
        scale = 1 / np.sqrt(embedding_dim)
        self.weight = Tensor(
            util.initialize((vocab_size, embedding_dim), scale=scale), name="E"
        )

    def forward(self, indices: Tensor) -> Tensor:
        return ops.lookup_rows(self.weight, indices)


def dot_product_rows(mat1: Tensor, mat2: Tensor) -> Tensor:
    """Given two Tensors of the same shape [batch_size, representation_size],
    compute the dot product _of each row_ of the two matrices.

    The output should be a Tensor that has shape [batch_size], with
    dot_product_rows(m1, m2)[i] being the dot product of row i of m1 with
    row i of m2.

    Args:
        mat1: Tensor of shape [batch_size, representation_size]
        mat2: Tensor of shape [batch_size, representation_size]

    Returns:
        Tensor of shape [batch_size], containing the row-wise dot products
    """
    # TODO: your code here
    # Hint: ops.sum_along_columns and ops.multiply are your friend :)
    products = ops.multiply(mat1, mat2)
    return ops.sum_along_columns(products)


class Word2Vec(nn.Module):
    def __init__(self, vocab: Vocabulary, embedding_dim: int = 100):
        super(Word2Vec, self).__init__()
        self.embeddings = Embedding(len(vocab), embedding_dim)
        self.context_embeddings = Embedding(len(vocab), embedding_dim)

    def forward(self, target_indices: Tensor, context_indices: Tensor) -> Tensor:
        """Forward pass of word2vec negative sampling model.

        Use the two Embedding layers defined in __init__ as well as the method
        dot_product_rows that you implemented above.

        Args:
            target_one_hots: [batch_size], indices of target words
            context_one_hots: [batch_size], indices of context words

        Returns:
            [batch_size] Tensor containing P(+ | w, c)
        """
        # TODO: implement!
        target_embeddings = self.embeddings(target_indices)
        context_embeddings = self.context_embeddings(context_indices)
        dot_product = dot_product_rows(target_embeddings, context_embeddings)
        return ops.sigmoid(dot_product)


# by convention, all positive examples come first in a batch, then negatives
def prepare_batch(
    positive_indices: list[tuple[int, int]],
    start: int,
    end: int,
    tokens: list[str],
    weights: list[int],
) -> dict[str, Tensor]:
    # lists of integers
    positive_targets, positive_contexts = zip(*positive_indices[start:end])

    negative_examples = []
    for index in range(start, end):
        negative_examples.extend(
            data.negatives_from_positive(
                tokens, weights, positive_examples[index], args.num_negatives
            )
        )
    negative_indices = data.examples_to_indices(negative_examples, vocab)
    # lists of integers
    negative_targets, negative_contexts = zip(*negative_indices)

    batch_targets = Tensor(np.array(positive_targets + negative_targets))
    batch_contexts = Tensor(np.array(positive_contexts + negative_contexts))

    # last batch might be different size
    batch_size = end - start
    # generate labels
    positive_labels = np.ones(batch_size)
    negative_labels = np.zeros(len(negative_examples))
    # [batch_size * (num_negatives + 1)]
    batch_labels = np.concatenate([positive_labels, negative_labels])

    return {
        "targets": batch_targets,
        "contexts": batch_contexts,
        "labels": Tensor(batch_labels),
    }


if __name__ == "__main__":

    # get command-line arguments
    args = util.get_args()

    # set random seed
    util.set_seed(args.seed)

    # build vocabulary
    vocab = Vocabulary.from_text_files([args.training_data], min_freq=args.min_freq)
    vocab_size = len(vocab)

    # scale frequencies for negative sampling
    vocab_weights = {
        token: vocab.frequencies[token] ** args.alpha for token in vocab.index_to_token
    }
    tokens = list(vocab_weights.keys())
    weights = list(vocab_weights.values())

    # list of (target word, context word) tuples
    positive_examples = data.generate_training_data(
        args.training_data, args.window_size, tokens
    )
    # list of (target word index, context word index)
    positive_indices = data.examples_to_indices(positive_examples, vocab)

    # initialize model
    model = Word2Vec(vocab, args.embedding_dim)
    # get optimizer
    optimizer = SGD(model.parameters(), lr=args.learning_rate)

    # main training loop
    batch_size = args.batch_size
    num_batches = int(len(positive_examples) / batch_size) + 1
    start_time = time.time()

    for epoch in range(args.num_epochs):
        # shuffle the data order
        random.shuffle(positive_indices)
        running_loss = 0.0
        for start in range(0, len(positive_indices), batch_size):
            # get batch to feed in to model
            end = min(len(positive_indices), start + batch_size)
            batch = prepare_batch(positive_indices, start, end, tokens, weights)

            # get P(+ | w, c) from model
            # [batch_size * (num_negatives + 1)]
            probabilities = model(batch["targets"], batch["contexts"])
            # compute loss
            loss = bce_loss(probabilities, batch["labels"])
            running_loss += loss.value

            # update the weights!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} average loss: {running_loss / num_batches}")

    print(f"Total training time: {time.time() - start_time}")

    if args.save_vectors:
        final_vectors = (
            model.embeddings.weight.value + model.context_embeddings.weight.value
        )
        util.save_vectors(tokens, final_vectors, args.save_vectors)
