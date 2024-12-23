from collections import Counter

import torch
import numpy as np

from edugrad.ops import reduce_mean
from edugrad.tensor import Tensor

import ops
from vocabulary import Vocabulary
from word2vec import bce_loss, dot_product_rows, Word2Vec


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TestOps:

    scalar = np.array(1.0)
    batch = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_log(self):
        # scalar forward test
        scalar = Tensor(TestOps.scalar)
        scalar_log = ops.log(scalar)
        np.testing.assert_allclose(scalar_log.value, np.log(TestOps.scalar))
        # scalar backward test
        scalar_log.backward()
        np.testing.assert_allclose(scalar.grad, 1.0)

        # batch forward test
        batch = Tensor(TestOps.batch)
        batch_log = ops.log(batch)
        np.testing.assert_allclose(batch_log.value, np.log(TestOps.batch))
        # batch backward test
        batch_mean = reduce_mean(batch_log)
        batch_mean.backward()
        batch_torch = torch.tensor(TestOps.batch, requires_grad=True)
        batch_torch.log().mean().backward()
        np.testing.assert_allclose(batch.grad, batch_torch.grad.numpy())

    def test_sigmoid(self):
        # scalar forward test
        scalar = Tensor(TestOps.scalar)
        sig_out = ops.sigmoid(scalar)
        sig_np = sigmoid(TestOps.scalar)
        np.testing.assert_allclose(sig_out.value, sig_np)
        # scalar backward test
        sig_out.backward()
        np.testing.assert_allclose(scalar.grad, 0.19661, rtol=1e-5)

        # batch forward test
        batch = Tensor(TestOps.batch)
        batch_sig = ops.sigmoid(batch)
        np.testing.assert_allclose(batch_sig.value, sigmoid(TestOps.batch))
        # batch backward test
        batch_mean = reduce_mean(batch_sig)
        batch_mean.backward()
        batch_torch = torch.tensor(TestOps.batch, requires_grad=True)
        batch_torch.sigmoid().mean().backward()
        np.testing.assert_allclose(batch.grad, batch_torch.grad.numpy())

    def test_multiply(self):
        # scalar forward test
        scalar = Tensor(TestOps.scalar)
        three = np.array(3.0)
        mul_out = ops.multiply(scalar, Tensor(three))
        np.testing.assert_allclose(mul_out.value, three)
        # scalar backward test
        mul_out.backward()
        np.testing.assert_allclose(scalar.grad, three)

        # batch forward test
        batch = Tensor(TestOps.batch)
        batch_mul = ops.multiply(batch, batch)
        np.testing.assert_allclose(batch_mul.value, TestOps.batch * TestOps.batch)
        # batch backward test
        batch_mean = reduce_mean(batch_mul)
        batch_mean.backward()
        batch_torch = torch.tensor(TestOps.batch, requires_grad=True)
        (batch_torch * batch_torch).mean().backward()
        np.testing.assert_allclose(batch.grad, batch_torch.grad.numpy())


class TestW2V:

    batch = np.array([[1.0, 2.0], [3.0, 4.0]])
    probs = np.array([0.6, 0.2, 0.7])
    labels = np.array([1, 0, 1])

    def test_dot(self):
        batch_tensor = Tensor(TestW2V.batch)
        dots = dot_product_rows(batch_tensor, batch_tensor)
        np.testing.assert_allclose(dots.value, np.array([5.0, 25.0]))

    def test_bce(self):
        loss = bce_loss(Tensor(TestW2V.probs), Tensor(TestW2V.labels))
        np.testing.assert_allclose(
            loss.value,
            torch.nn.functional.binary_cross_entropy(
                torch.Tensor(TestW2V.probs), torch.Tensor(TestW2V.labels)
            ),
        )

    def test_forward(self):
        vocab = Vocabulary(Counter(["a", "b"]), special_tokens=[])
        model = Word2Vec(vocab, 2)
        model.embeddings.weight = Tensor(np.eye(2))
        model.context_embeddings.weight = Tensor(np.eye(2))
        target_words = [0, 1, 0]
        context_words = [1, 1, 0]
        probabilities = model(
            Tensor(np.array(target_words)), Tensor(np.array(context_words))
        )
        np.testing.assert_allclose(
            probabilities.value, np.array([0.5, 0.73106, 0.73106]), rtol=1e-5
        )
