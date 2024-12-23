from typing import Any, Callable

import numpy as np
from vocabulary import Vocabulary

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocab

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)



class SSTClassificationDataset(Dataset):

    labels_to_string = {0: "terrible", 1: "bad", 2: "so-so", 3: "good", 4: "excellent"}
    label_one_hots = np.eye(len(labels_to_string))

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        example = self.__getitem__(index)
        # TODO (~3 lines): build the bag of words vector for one example, stored in the variable above, here!
        # Note: use the self.vocab Vocabulary object to get integer indices
        # bag_of_words should be a 1-D numpy array of shape [vocabulary_size]
        # element i of this vector should be how many times word i (where i is the index in the Vocabulary)
        # occurred in the example
        bag_of_words = np.zeros(len(self.vocab))
        for word in example['review']:
            bag_of_words[self.vocab[word]] += 1
        return {
            "review": bag_of_words,
            "label": SSTClassificationDataset.label_one_hots[example["label"]],
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        examples = [self.example_to_tensors(index) for index in range(start, end)]
        return {
            "review": np.stack([example["review"] for example in examples]),
            "label": np.stack([example["label"] for example in examples]),
        }

    @classmethod
    def from_files(cls, reviews_file: str, labels_file: str, vocab: Vocabulary = None):
        with open(reviews_file, "r") as reviews, open(labels_file, "r") as labels:
            review_lines = reviews.readlines()
            label_lines = labels.readlines()
        examples = [
            {
                # review is text, stored as a list of tokens
                "review": review_lines[line].strip("\n").split(" "),
                "label": int(label_lines[line].strip("\n")),
            }
            for line in range(len(review_lines)) 
        ]
        # initialize a vocabulary from the reviews, if none is given
        if not vocab:
            vocab = Vocabulary.from_text_files([reviews_file])
        return cls(examples, vocab)
