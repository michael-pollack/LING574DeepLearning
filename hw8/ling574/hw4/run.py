import argparse
import numpy as np

from edugrad.data import BatchIterator, Batch
from edugrad.ops import reduce_sum
from optim import SGD, Adagrad
from edugrad.tensor import Tensor
from data import SSTClassificationDataset
from model import DeepAveragingNetwork
from ops import cross_entropy_loss, softmax_rows, multiply


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Computes accuracy of a set of predictions.

    Args:
        probabilities: [batch_size, num_labels], model predictions
        labels: [batch_size, num_labels], one hots for gold labels

    Returns:
        percentage of correct predictions, where model prediction is the
        argmax of its probabilities
    """
    # [batch_size]
    predictions = probabilities.argmax(axis=1)
    # [batch_size]
    labels = labels.argmax(axis=1)
    return (predictions == labels).astype(int).mean()


if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=575)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--word_dropout", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument(
        "--train_reviews",
        type=str,
        default="/dropbox/20-21/575k/data/sst/train-reviews.txt",
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        default="/dropbox/20-21/575k/data/sst/train-labels.txt",
    )
    parser.add_argument(
        "--dev_reviews",
        type=str,
        default="/dropbox/20-21/575k/data/sst/dev-reviews.txt",
    )
    parser.add_argument(
        "--dev_labels", type=str, default="/dropbox/20-21/575k/data/sst/dev-labels.txt"
    )
    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)

    # build datasets
    sst_train = SSTClassificationDataset.from_files(
        args.train_reviews, args.train_labels
    )
    sst_dev = SSTClassificationDataset.from_files(
        args.dev_reviews, args.dev_labels, vocab=sst_train.vocab
    )
    # data as np arrays
    training_data = sst_train.batch_as_tensors(0, len(sst_train))
    dev_data = sst_dev.batch_as_tensors(0, len(sst_dev))
    dev_lengths = Tensor(dev_data["review"].sum(axis=1))
    # batch iterator for training data
    training_iterator = BatchIterator(args.batch_size)

    # build model
    dan = DeepAveragingNetwork(
        # vocab size
        len(sst_train.vocab),
        args.embedding_dim,
        args.hidden_dim,
        # number of labels
        len(SSTClassificationDataset.labels_to_string),
    )
    optimizer = Adagrad(dan.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        batch_num = 0
        running_loss = 0.0
        for batch in training_iterator(training_data["review"], training_data["label"]):
            batch_num += 1

            # apply word dropout if specified in arguments
            if args.word_dropout:
                words = batch.inputs.value
                labels = batch.targets.value
                # get a mask of words to drop out
                # 1 = keep, 0 = drop
                # [batch_size, vocab_size]
                drop_mask = np.random.binomial(1, 1 - args.word_dropout, size=words.shape)
                words = words * drop_mask
                # dropout can cause length-zero inputs, so keep only the non-zero length ones
                # [batch_size]
                lengths = words.sum(axis=1)
                nonzero = lengths != 0
                words = words[nonzero]
                labels = labels[nonzero]
                # repackage as Batch of Tensors, necessary to make grad have the right shape
                batch = Batch(Tensor(words, name="x"), Tensor(labels, name="y"))

            # get probabilities and loss
            lengths = Tensor(batch.inputs.value.sum(axis=1))
            logits = dan(batch.inputs, lengths)
            probabilities = softmax_rows(logits)
            loss = cross_entropy_loss(probabilities, batch.targets)
            running_loss += loss.value

            # add L2 regularization if asked for
            if args.l2:
                l2_sum = Tensor(np.array(0.0))
                for parameter in dan.parameters():
                    l2_sum += reduce_sum(parameter ** 2)
                l2_sum = multiply(Tensor(np.array(args.l2)), l2_sum)
                loss += l2_sum

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} train loss: {running_loss / batch_num}")

        # get dev loss every epoch
        probabilities = softmax_rows(dan(Tensor(dev_data["review"]), dev_lengths))
        epoch_loss = cross_entropy_loss(probabilities, Tensor(dev_data["label"]))
        print(f"Epoch {epoch} dev loss: {epoch_loss.value}")

    # get dev accuracy at the very end
    probabilities = softmax_rows(dan(Tensor(dev_data["review"]), dev_lengths))
    dev_accuracy = accuracy(probabilities.value, dev_data["label"])
    print(f"Final dev accuracy: {dev_accuracy}")