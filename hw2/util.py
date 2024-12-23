import argparse
import random
import numpy as np
from typing import Iterable, Any


def set_seed(seed: int) -> None:
    """Sets various random seeds. """
    random.seed(seed)
    np.random.seed(seed)


def initialize(shape: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    """Initialize a weight matrix with random uniform values in [-scale, scale)

    Args:
        shape: tuple containing desired shape
        scale: absolute value of upper / lower bound

    Returns:
        matrix containing random values
    """
    return (2 * np.random.random(shape) - 1) * scale


def vector_to_string(vec: Iterable, delimiter: str = "\t") -> str:
    """String representation of a vector, for writing to a file.

    Args:
        vec: assumed to be a 1-D numpy array, but can be any iterable
        delimiter: what to separate the entries of the vector buy

    Returns:
        string representation of a vector
    """
    return delimiter.join(str(element) for element in vec)


def save_vectors(
    tokens: list[str], embeddings: np.ndarray, filename: str, delimiter: str = "\t"
) -> None:
    """Write emeddings to a file.

    Args:
        tokens: list of tokens corresponding to the embeddings
        embeddings: (vocab_size, embedding_dim) array of vectors
        filename: file to write to
        delimiter: what to separate entries by
    """
    with open(filename, "w") as f:
        for index in range(len(tokens)):
            f.write(
                f"{tokens[index]}{delimiter}{vector_to_string(embeddings[index])}\n"
            )


def read_vectors(filename: str, delimiter: str = "\t") -> dict[str, np.ndarray]:
    vectors = {}
    with open(filename, "r") as f:
        for line in f:
            split = line.strip("\n").split(delimiter)
            vectors[split[0]] = np.array(split[1:]).astype(float)
    return vectors


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Specify the random seed.")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="Dimension of the word embeddings.",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=5,
        help="Number of negative samples per positive sample.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Exponent for scaling counts for taking negative samples.",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="Only include tokens which occur at least this often.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2,
        help="How many words before/after a target word to use as data.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="How many passes through the training data.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="How many positive examples per batch"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for SGD."
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default="/dropbox/23-24/574/data/sst/train-reviews.txt",
        help="Path to file containing raw text.",
    )
    parser.add_argument(
        "--save_vectors",
        type=str,
        default=None,
        help="If specified, vectors will be saved to this file as plain text.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="If specified, plot will be saved to this file as png.",
    )
    args = parser.parse_args()
    return args
