import argparse
import copy
import random

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from data import Seq2SeqDataset
from model import Seq2SeqModel


if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=575)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--temp", type=float, default=4.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.125)
    parser.add_argument("--generate_every", type=int, default=1)
    parser.add_argument(
        "--train_source",
        type=str,
        default="/dropbox/20-21/575k/data/europarl-v7-es-en/validation.en.txt",
    )
    parser.add_argument(
        "--train_target",
        type=str,
        default="/dropbox/20-21/575k/data/europarl-v7-es-en/validation.es.txt",
    )
    parser.add_argument(
        "--dev_source",
        type=str,
        default="/dropbox/20-21/575k/data/europarl-v7-es-en/validation.en.txt",
    )
    parser.add_argument(
        "--dev_target",
        type=str,
        default="/dropbox/20-21/575k/data/europarl-v7-es-en/validation.es.txt",
    )
    parser.add_argument(
        "--test_source",
        type=str,
        default="/dropbox/20-21/575k/data/europarl-v7-es-en/test.en.txt"
    )
    parser.add_argument(
        "--test_target",
        type=str,
        default="/dropbox/20-21/575k/data/europarl-v7-es-en/test.es.txt"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test.en.txt.es"
    )
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build datasets
    train_set = Seq2SeqDataset.from_files(args.train_source, args.train_target)
    dev_set = Seq2SeqDataset.from_files(args.dev_source, args.dev_target, vocab=train_set.vocab)
    test_set = Seq2SeqDataset.from_files(args.test_source, args.test_target, vocab=train_set.vocab)
    # arrays of entire dev set
    dev_data = dev_set.batch_as_tensors(0, len(dev_set))
    test_data = test_set.batch_as_tensors(0, len(test_set))
    # convert to Tensors
    dev_data = {key: torch.LongTensor(value) for key, value in dev_data.items()}
    test_data = {key: torch.LongTensor(value) for key, value in test_data.items()}
    
    # build model
    padding_index = train_set.vocab[Seq2SeqDataset.PAD]
    bos_index = train_set.vocab[Seq2SeqDataset.BOS]
    eos_index = train_set.vocab[Seq2SeqDataset.EOS]
    # get the language model
    model = Seq2SeqModel(
        args.embedding_dim,
        args.hidden_dim,
        args.num_layers,
        len(train_set.vocab),
        padding_index,
        args.dropout
    )

    print("Example translations before training:")
    for idx in range(8):
        input_sentence = dev_data['source'][idx, :]
        input_tensor = input_sentence.unsqueeze(0)
        input_length = dev_data['lengths'][0][idx]
        input_plaintext = "".join(train_set.vocab.indices_to_tokens(input_sentence.numpy()[1:input_length]))
        output_plaintext = model.generate(
            input_tensor,
            [input_length],
            bos_index,
            eos_index,
            train_set.vocab,
            args.temp
        )
        print(f"INPUT: {input_plaintext}")
        print(f"GENERATED OUTPUT: {output_plaintext}")
    print("")

    # get training things set up
    data_size = len(train_set)
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
            batch = train_set.batch_as_tensors(
                start, min(start + batch_size, data_size)
            )
            model.train()
            # get probabilities and loss
            # [batch_size, max_seq_len, vocab_size]
            logits = model(
                torch.LongTensor(batch['source']),
                torch.LongTensor(batch['target_x']),
                batch['lengths']
            )
            # transpose for torch cross entropy format
            # [batch_size, vocab_size, max_seq_len]
            logits = logits.transpose(1, 2)
            # [batch_size, max_seq_len]
            loss = torch.nn.functional.cross_entropy(
                logits,
                # batch["target"]: [batch_size, max_seq_len]
                torch.LongTensor(batch['target_y']),
                reduction="mean",
                ignore_index=padding_index
            )
            running_loss += loss.item()

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = round(running_loss / len(starts), 4)
        print(f"Epoch {epoch} train loss: {train_loss}")

        # get dev loss every epoch
        model.eval()
        # [batch_size, max_seq_len, vocab_size]
        logits = model(dev_data['source'], dev_data['target_x'], dev_data['lengths'])
        epoch_loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            torch.LongTensor(dev_data['target_y']),
            reduction="mean",
            ignore_index=padding_index
        )
        print(
            f"Epoch {epoch} dev loss: {round(epoch_loss.item(), 4)};"
            f" perplexity (nats): {round(epoch_loss.exp().item(), 4)}"
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New best loss; saving current model")
            best_model = copy.deepcopy(model)

        # Generate some text every N epochs
        if (epoch + 1) % args.generate_every == 0:
            print("Generating example output:")
            for idx in range(8):
                input_sentence = dev_data['source'][idx, :]
                input_tensor = input_sentence.unsqueeze(0)
                input_length = dev_data['lengths'][0][idx]
                input_plaintext = "".join(train_set.vocab.indices_to_tokens(input_sentence.numpy()[1:input_length]))
                output_plaintext = model.generate(
                    input_tensor,
                    [input_length],
                    bos_index,
                    eos_index,
                    train_set.vocab,
                    args.temp
                )
                print(f"INPUT: {input_plaintext}")
                print(f"GENERATED OUTPUT: {output_plaintext}")
        print("")

    print("Printing test generations to output file")
    with open(args.output_file, 'w') as fout:
        for idx in tqdm(range(len(test_set))):
            input_sentence = test_data['source'][idx, :]
            input_tensor = input_sentence.unsqueeze(0)
            input_length = test_data['lengths'][0][idx]
            output_plaintext = model.generate(
                input_tensor,
                [input_length],
                bos_index,
                eos_index,
                train_set.vocab,
                args.temp
            )
            print(output_plaintext, file=fout)

    print("Done")
