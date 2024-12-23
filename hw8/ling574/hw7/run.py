import argparse
import copy
import random

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from data import SSTLanguageModelingDataset
from model import LSTMLanguageModel
from vocabulary import Vocabulary


def get_mask(characters: np.ndarray, padding_index: int) -> np.ndarray:
    """Generate a mask, where 1 corresponds to true predicted characters, and 0 to the padding token.

    Example usage:
    >>> get_mask(np.array([[1, 3, 3], [1, 4, 3], [1, 4, 5]]), 3)
    >>> np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

    Arguments:
        characters: [batch_size, max_seq_len]
            indices of characters in a batch of sequences
        padding_index: the index of the padding token

    Returns:
        [batch_size, max_seq_len] numpy array, as specified above
    """
    # TODO: implement here (~1-2 lines)
    return np.array([[0 if i == padding_index else 1 for i in seq] for seq in characters])


def mask_loss(loss: Tensor, target: np.ndarray, padding_index: int) -> Tensor:
    """Given per-token losses across a batch of sequences, this method masks out the
    losses for padding tokens and then returns the mean loss.

    Note: the denominator in the mean should be the number of real/true tokens, i.e.
    the number of non-padded tokens.  In other words, the output should be:

    1 / N * sum(masked_loss)

    where:
        masked_loss is the same as loss, but with zeros wherever padding_index occurs as a target
        N is the total number of elements of loss that _do not_ correspond to padding_index

    Arguments:
        loss: [batch_size, max_seq_len] Tensor
            -log P(c_i | c_<i) for each character in each of a batch of sequences
        target: [batch_size, max_seq_len] numpy array 
            the (padded) target character indices for the batch of sequences
        padding_index: the integer index of the padding token
            this is the index which should be masked out

    Returns:
        Tensor of shape (), i.e. one float
        Containing the mean of loss, after masking out the losses corresponding to padding_index
    """
    # TODO: implement here! (~4-5 lines)
    # Hint: use get_mask first; t.sum() on a torch Tensor t will return the sum of its elements
    mask = torch.tensor(get_mask(target, padding_index))
    masked_loss = mask * loss
    unmasked_loss = torch.sum(masked_loss)
    return 1 / torch.sum(mask) * unmasked_loss


def generate(
    model: LSTMLanguageModel,
    bos_index: int,
    batch_size: int,
    max_len: int,
    vocab: Vocabulary,
    temp: float = 3.0,
) -> list[str]:
    """Generate text from an LSTM Language Model, by iteratively sampling.

    Note: this generates a _batch_ of texts, all of the same length.  This will keep sampling
    even after some sequences have generated </s>.

    Arguments:
        model: the model to use for generation
        bos_index: integer index of <s>
        batch_size: how many texts to generate
        max_len: how many characters to generate
        vocab: the Vocabulary of the model
        temp: softmax temperature; higher makes the distribution closer to argmax

    Returns:
        list of strings; batch_size long, each one is max_len characters
    """
    generated = torch.zeros((batch_size, 1), dtype=torch.int64) + bos_index
    with torch.no_grad():
        for idx in range(max_len):
            # [batch_size, seq_len, vocab_size]
            logits = model(generated, [idx + 1] * batch_size)
            # [batch_size, 1, vocab_size]
            logits = logits[:, -1, :].unsqueeze(dim=1) * temp
            probs = torch.distributions.categorical.Categorical(logits=logits)
            # get next character
            # [batch_size, 1]
            next_chars = probs.sample()
            # append to the sequence length dimension
            # [batch_size, seq_len + 1]
            generated = torch.cat([generated, next_chars], dim=1)
    texts = generated.numpy()
    texts = ["".join(vocab.indices_to_tokens(text)) for text in texts]
    return texts


if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=575)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=60)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--generate_every", type=int, default=4)
    parser.add_argument("--generate_length", type=int, default=50)
    parser.add_argument("--num_generate", type=int, default=10)
    parser.add_argument("--temp", type=float, default=2.5)
    parser.add_argument(
        "--train_data",
        type=str,
        default="/dropbox/20-21/575k/data/sst/train-reviews.txt",
    )
    parser.add_argument(
        "--dev_data",
        type=str,
        default="/dropbox/20-21/575k/data/sst/dev-reviews.txt",
    )
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build datasets
    sst_train = SSTLanguageModelingDataset.from_file(args.train_data)
    sst_dev = SSTLanguageModelingDataset.from_file(args.dev_data, vocab=sst_train.vocab)
    # arrays of entire dev set
    dev_data = sst_dev.batch_as_tensors(0, len(sst_dev))
    # convert to Tensors
    dev_data = {key: torch.LongTensor(value) for key, value in dev_data.items()}

    # build model
    padding_index = sst_train.vocab[SSTLanguageModelingDataset.PAD]
    # get the language model
    model = LSTMLanguageModel(
        args.embedding_dim,
        args.hidden_dim,
        len(sst_train.vocab),
        padding_index,
        args.dropout,
    )

    # get training things set up
    data_size = len(sst_train)
    batch_size = args.batch_size
    starts = list(range(0, data_size, batch_size))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2, lr=args.lr)
    best_loss = float("inf")
    best_model = None

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        # shuffle batches
        random.shuffle(starts)
        for start in tqdm(starts):
            batch = sst_train.batch_as_tensors(
                start, min(start + batch_size, data_size)
            )
            model.train()
            # get probabilities and loss
            # [batch_size, max_seq_len, vocab_size]
            logits = model(
                torch.LongTensor(batch["text"]), torch.LongTensor(batch["length"])
            )
            # transpose for torch cross entropy format
            # [batch_size, vocab_size, max_seq_len]
            logits = logits.transpose(1, 2)
            # [batch_size, max_seq_len]
            all_loss = torch.nn.functional.cross_entropy(
                logits,
                # batch["target"]: [batch_size, max_seq_len]
                torch.LongTensor(batch["target"]),
                reduction="none",
            )
            # mask out the PAD symbols in the loss
            loss = mask_loss(all_loss, batch["target"], padding_index)

            running_loss += loss.item()

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} train loss: {running_loss / len(starts)}")

        # get dev loss every epoch
        model.eval()
        # [batch_size, max_seq_len, vocab_size]
        logits = model(dev_data["text"], dev_data["length"])
        epoch_loss = mask_loss(
            torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), dev_data["target"], reduction="none"
            ),
            dev_data["target"].numpy(),
            padding_index,
        )
        print(f"Epoch {epoch} dev loss: {epoch_loss.item()}; perplexity (nats): {epoch_loss.exp()}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New best loss; saving current model")
            best_model = copy.deepcopy(model)

        # generate some text every N epochs
        if (epoch + 1) % args.generate_every == 0:
            print(
                generate(
                    model,
                    sst_train.vocab[SSTLanguageModelingDataset.BOS],
                    args.num_generate,
                    args.generate_length,
                    sst_train.vocab,
                    temp=args.temp,
                )
            )
