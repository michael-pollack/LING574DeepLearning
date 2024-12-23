from collections import Counter

import torch
import numpy as np
from edugrad.tensor import Tensor

import data
import model
import ops
import optim
from vocabulary import Vocabulary


class TestDataAndModel:

    example = {"review": ["the", "movie", "was", "the", "best"], "label": 4}
    vocab = Vocabulary(Counter(example["review"]))
    # NB: not how we're defining datasets in the main running script
    dataset = data.SSTClassificationDataset([example], vocab)

    def test_bag_of_words(self):
        tensors = TestDataAndModel.dataset.example_to_tensors(0)
        np.testing.assert_allclose(tensors["label"], np.array([0, 0, 0, 0, 1]))
        np.testing.assert_allclose(tensors["review"], np.array([0, 2, 1, 1, 1]))

    def test_forward(self):
        # set vocab_size, embedding_size, hidden_size, output_size all to 2
        dan = model.DeepAveragingNetwork(
            2,
            2,
            2,
            2,
        )
        # manually set all of the parameters :scream:
        dan.embeddings.weight = Tensor(np.eye(2))
        dan.hidden1.weight = Tensor(np.eye(2))
        dan.hidden1.bias = Tensor(np.zeros(2))
        dan.hidden2.weight = Tensor(np.eye(2))
        dan.hidden2.bias = Tensor(np.zeros(2))
        dan.output.weight = Tensor(np.eye(2))
        dan.output.bias = Tensor(np.zeros(2))
        # get outputs
        outputs = dan(
            # bags of words
            Tensor(np.array([[2.0, 3.0], [3.0, 1.0]])),
            # lengths
            Tensor(np.array([5.0, 4.0])),
        )
        np.testing.assert_allclose(outputs.value, np.array([[0.4, 0.6], [0.75, 0.25]]))


class TestOps:

    batch = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([1, 0])

    def test_exp(self):
        # test forward
        base = Tensor(TestOps.batch)
        exps = ops.exp(base)
        np.testing.assert_allclose(
            exps.value, np.array([[2.71828, 7.38905], [20.08554, 54.59815]]), rtol=1e-5
        )
        # test backward
        mean = ops.reduce_mean(exps)
        mean.backward()
        torch_base = torch.tensor(TestOps.batch, requires_grad=True)
        torch_base.exp().mean().backward()
        np.testing.assert_allclose(base.grad, torch_base.grad.numpy())

    def test_softmax(self):
        softmax_out = ops.softmax_rows(Tensor(TestOps.batch))
        softmax_torch = torch.nn.functional.softmax(torch.Tensor(TestOps.batch), dim=1)
        np.testing.assert_allclose(softmax_out.value, softmax_torch.numpy())

    def test_cross_entropy(self):
        # [batch_size, num_labels]
        label_one_hots = np.eye(TestOps.batch.shape[1])[TestOps.labels]
        # [batch_size, num_labels]
        probabilities = ops.softmax_rows(Tensor(TestOps.batch))
        # [], i.e. scalar
        cross_ent = ops.cross_entropy_loss(probabilities, Tensor(label_one_hots))
        torch_cross_ent = torch.nn.functional.cross_entropy(
            # NB: torch wants logits, not probabilities
            # and integer indices, not one hot vectors
            torch.Tensor(TestOps.batch),
            torch.LongTensor(TestOps.labels),
        )
        np.testing.assert_allclose(cross_ent.value, torch_cross_ent.numpy())


class TestAdagrad:

    param = Tensor(np.array([1.0, 2.0]))
    optimizer = optim.Adagrad([param])

    def test_adagrad(self):
        TestAdagrad.optimizer._eps = 0.0
        TestAdagrad.param.grad = np.array([2.0, 3.0])
        TestAdagrad.optimizer.step()
        np.testing.assert_allclose(TestAdagrad.param.value, np.array([0.99, 1.99]))