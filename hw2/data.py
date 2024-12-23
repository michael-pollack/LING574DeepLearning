import random
from vocabulary import Vocabulary


def negative_samples(
    tokens: list[str], weights: list[float], num_samples: int = 1, avoid: str = None
) -> list[str]:
    """Get negative samples: random tokens, weighted by weights, and avoiding the string `avoid', which
    will be the true context token for a given data point.

    Args:
        tokens: list to choose from
        weights: weights for the choice
        num_samples: how many samples to draw
        avoid: the token, if any, to avoid

    Returns:
        list of sampled tokens
    """
    # TODO (~3-5 lines): implement this method
    if avoid is not None:
        chosen_tokens, chosen_weights = zip(*[(token, weight) for token, weight in zip(tokens, weights) if token != avoid])
    else:
        chosen_tokens, chosen_weights = tokens, weights
    return random.choices(chosen_tokens, weights=chosen_weights, k=num_samples)


def negatives_from_positive(
    tokens: list[str],
    weights: list[float],
    positive_sample: tuple[str, str],
    num_negatives: int,
) -> list[tuple[str, str]]:
    """Generates negative samples from a given positive sample.

    Args:
        tokens: tokens to sample from
        weights: weights for the tokens
        positive_sample: the true (target word, context word) pair
        num_negatives: how many negative samples to generate

    Returns:
        list of num_negatives pairs of (target word, negative word) pairs
    """
    return [
        (positive_sample[0], negative)
        for negative in negative_samples(
            # avoid the true target word
            tokens,
            weights,
            num_samples=num_negatives,
            avoid=positive_sample[1],
        )
    ]


def get_positive_samples(
    text: list[str], window_size: int, tokens: list[str]
) -> list[tuple[str, str]]:
    """Iterate through a text, generating positive skip-gram examples.

    Args:
        text: list of tokens
        window_size: how far on either side of each token to look
        tokens: list of tokens in the vocabulary; only include tokens from this list in positive samples

    Returns:
        a list of (target_word, context_word) tuples
    """
    # TODO (~7-10 lines): implement this method
    positive_samples = []
    vocab = set(tokens)
    for i in range(len(text)):
        start = max(0, (i - window_size))
        end = min(len(text), i + window_size + 1)
        if text[i] in vocab:
            for j in range(start, end):
                if (j != i) and (text[j] in vocab):
                    positive_samples.append((text[i], text[j]))
    return positive_samples


def generate_training_data(
    filename: str, window_size: int, tokens: list[str]
) -> list[tuple[str, str]]:
    """Read a raw text file and generate positive samples.

    Args:
        filename: file to read; each line will be passed to `get_positive_samples`
        window_size: how many tokens before/after to use
        tokens: list of tokens in the vocabulary

    Returns:
        list of positive (target word, context word) pairs
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.extend(get_positive_samples(line.strip("\n").split(), window_size, tokens))
    return data


def examples_to_indices(
    examples: list[tuple[str, str]], vocab: Vocabulary
) -> list[tuple[int, int]]:
    """Converts a list of examples of pairs of tokens into the corresponding indices
    according to the given Vocabulary.

    Args:
        examples: list of (token, token) pairs
        vocab: Vocabulary to use for token --> index mapping

    Returns:
        list of (index, index) pairs
    """
    # zip(*...) "unzips" the list of tuples into a tuple of lists
    targets, contexts = zip(*examples)
    target_indices = vocab.tokens_to_indices(targets)
    context_indices = vocab.tokens_to_indices(contexts)
    # zip back together to get the right pairs
    return list(zip(target_indices, context_indices))