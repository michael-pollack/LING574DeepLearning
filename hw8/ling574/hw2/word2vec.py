import argparse
import random
import time
from typing import Any, Iterable

import numpy as np

import data
import util
from vocabulary import Vocabulary


# type hint for a model output
Result = dict[str, Any]


def sigmoid(x: float) -> float:
    """Returns sigmoid(x) """
    return 1 / (1 + np.exp(-x))


class SGNS:
    def __init__(self, vocab_size: int, embedding_dim: int):
        scale = 1 / np.sqrt(embedding_dim)
        self.embeddings: np.ndarray = util.initialize(
            (vocab_size, embedding_dim), scale=scale
        )
        self.context_embeddings: np.ndarray = util.initialize(
            (vocab_size, embedding_dim), scale=scale
        )

    def forward(self, example: tuple[int, int]) -> Result:
        """Do the forward pass of the word2vec model for a single example
        of (target word, context word) pair.

        This method will use the variables `self.embeddings` and `self.context_embeddings`
        defined just above.

        Args:
            example: a pair of integers, word IDs in the vocab for the example (w, c)

        Returns:
            a dictionary, with the following entries:
                target_word_embedding: 1-D numpy array; the word embedding u_w for target word
                context_word_embedding: 1-D numpy array; the context embedding c_w' for context word
                probability: float, the probability P(+ | w, c)
        """
        # TODO (~5 lines): implement this method, returning the relevant values in the dictionary below
        w, c = example
        u_w = self.embeddings[w]
        c_w = self.context_embeddings[c]
        dot_product = np.dot(u_w, c_w)
        probability = sigmoid(dot_product)
        return {
            "target_word_embedding": u_w,
            "context_word_embedding": c_w,
            "probability": probability,
        }


def get_positive_context_gradient(
    positive_result: Result, negative_results: Iterable[Result]
) -> np.ndarray:
    """Compute dL / dC_pos where C_pos is the context word embedding for the
    context word of the positive example.

    Args:
        positive_result: the Result from forward pass with (w, c_+)
        negative_result: the Results from all forward passes with (w, c_-i)

    Return:
        the gradient, an array of shape [embedding_dim]
    """
    # TODO (~1-2 lines): implement
    return (positive_result["probability"] - 1) * positive_result["target_word_embedding"]


def get_negative_context_gradients(
    positive_result: Result, negative_results: Iterable[Result]
) -> list[np.ndarray]:
    """Compute dL / dC_-i where C_-i is the context word embedding for the
    context word of the i'th negative example.

    Args:
        positive_result: the Result from forward pass with (w, c_+)
        negative_result: the Results from all forward passes with (w, c_-i)

    Return:
        the gradients, a list of arrays, each of shape [embedding_dim]
    """
    # TODO (~2-4 lines): implement
    gradients = [result["probability"] * result["target_word_embedding"] for result in negative_results]
    return gradients


def get_target_word_gradient(
    positive_result: Result, negative_results: Iterable[Result]
) -> np.ndarray:
    """Compute dL / dw, where w is the target word embedding.

    Args:
        positive_result: the Result from forward pass with (w, c_+)
        negative_result: the Results from all forward passes with (w, c_-i)

    Return:
        the gradient, an array of shape [embedding_dim]
    """
    # TODO (~4-8 lines): implement
    pos_gradient = (positive_result["probability"] - 1) * positive_result["context_word_embedding"]
    neg_gradients = [result["probability"] * result["context_word_embedding"] for result in negative_results]
    neg_gradient_sum = np.sum(neg_gradients, axis=0)
    return pos_gradient + neg_gradient_sum


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
    positive_indices = data.examples_to_indices(positive_examples, vocab)

    # initialize the model
    model = SGNS(vocab_size, args.embedding_dim)

    data_order = list(range(len(positive_examples)))
    learning_rate = args.learning_rate

    start_time = time.time()

    # main training loop
    for epoch in range(args.num_epochs):

        # shuffle the data indices
        random.shuffle(data_order)

        for data_index in data_order:

            # get indices for positive example
            positive_example = positive_indices[data_index]
            # get negative examples and their indices
            negative_examples = data.negatives_from_positive(
                tokens, weights, positive_examples[data_index], args.num_negatives
            )
            negatives = data.examples_to_indices(negative_examples, vocab)

            # forward passes
            positive_result = model.forward(positive_example)
            negative_results = [
                model.forward(negative_example) for negative_example in negatives
            ]

            # compute all gradients
            # dL / dc_+ for the positive context word
            positive_context_gradient = get_positive_context_gradient(
                positive_result, negative_results
            )

            # dL / dc_-i for each negative context word
            negative_context_gradients = get_negative_context_gradients(
                positive_result, negative_results
            )

            # dL / dw for the target word w
            target_gradient = get_target_word_gradient(
                positive_result, negative_results
            )

            # update parameters
            # target word
            model.embeddings[positive_example[0]] -= learning_rate * target_gradient
            # positive context word
            model.context_embeddings[positive_example[1]] -= (
                learning_rate * positive_context_gradient
            )
            # negative context words
            for negative_index in range(len(negatives)):
                model.context_embeddings[negatives[negative_index][1]] -= (
                    learning_rate * negative_context_gradients[negative_index]
                )

    print(f"Total training time: {time.time() - start_time}")

    if args.save_vectors:
        final_vectors = model.embeddings + model.context_embeddings
        util.save_vectors(tokens, final_vectors, args.save_vectors)
