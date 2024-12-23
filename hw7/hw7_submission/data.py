from collections import Counter
from typing import Any, Callable

import numpy as np
from hw6_data import pad_batch
from vocabulary import Vocabulary

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocab
        self.num_labels = len(self.vocab)
        self._label_one_hots = np.eye(self.num_labels)

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


def example_from_characters(characters: list[str], bos: str, eos: str) -> Example:
    """Generate a sequence of language modeling targets from a list of characters.

    Example usage:
    >>> example_from_characters(['a', 'b', 'c'], '<s>', '</s>')
    >>> {'text': ['<s>', 'a', 'b', 'c'], 'target': ['a', 'b', 'c', '</s>'], 'length': 4}

    Arguments:
        characters: a list of strings, the characters in a sequence
        bos: beginning of sequence symbol, to be prepended as an input
        eos: end of sequence symbol, to be appended as a target

    Returns:
        an Example dictionary, as given in the example above, with three fields:
        text, target, and length
    """
    # TODO: implement here (~2-3 lines)
    return {
        'text': [bos] + characters,
        'target': characters + [eos],
        'length': len(characters) + 1
    }


class SSTLanguageModelingDataset(Dataset):

    BOS = "<s>"
    EOS = "</s>"
    PAD = "<PAD>"

    def example_to_indices(self, index: int) -> dict[str, Any]:
        example = self.__getitem__(index)
        return {
            "text": np.array(self.vocab.tokens_to_indices(example["text"])),
            "target": self.vocab.tokens_to_indices(example["target"]),
            "length": example["length"],
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, Any]:
        examples = [self.example_to_indices(index) for index in range(start, end)]
        padding_index = self.vocab[SSTLanguageModelingDataset.PAD]
        # pad texts to [batch_size, max_seq_len] array
        texts = pad_batch([np.array(example["text"]) for example in examples], padding_index)
        # target: [batch_size, max_seq_len], indices for next character
        target = pad_batch([np.array(example["target"]) for example in examples], padding_index)
        return {
            "text": texts,
            "target": target,
            "length": [example["length"] for example in examples],
        }

    @classmethod
    def from_file(cls, text_file: str, vocab: Vocabulary = None):
        examples = []
        counter: Counter = Counter()
        with open(text_file, "r") as reviews:
            for line in reviews:
                string = line.strip("\n")
                counter.update(string)
                # generate example from line
                examples.append(
                    example_from_characters(
                        list(string),
                        SSTLanguageModelingDataset.BOS,
                        SSTLanguageModelingDataset.EOS,
                    )
                )
        if not vocab:
            vocab = Vocabulary(
                counter,
                special_tokens=(
                    Vocabulary.UNK,
                    SSTLanguageModelingDataset.BOS,
                    SSTLanguageModelingDataset.EOS,
                    SSTLanguageModelingDataset.PAD,
                ),
            )
        return cls(examples, vocab)
