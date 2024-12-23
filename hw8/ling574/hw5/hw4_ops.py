import numpy as np
from edugrad.ops import Operation, tensor_op, reduce_mean, relu
from edugrad.tensor import Tensor

from hw3_ops import sum_along_columns, log, multiply


@tensor_op
class divide(Operation):
    """Divide row-wise a [batch_size, dimension] Tensor by a [batch-size]
    Tensor of scalars.

    Example:
        a = Tensor(np.array([[1., 2.], [3., 4.]]))
        b = Tensor(np.array([2., 3.]))
        divide(a, b).value == np.array([[0.5, 1.0], [1.0, 1.33]])
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.append(a)
        ctx.append(b)
        # broadcast b to [batch_size, 1]
        return a / b[:, np.newaxis]

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx[-2:]
        # broadcast b to a column
        b_column = b[:, np.newaxis]
        # a gradients
        inv_b_column = 1 / b_column
        num_columns = a.shape[1]
        broadcast = np.hstack([inv_b_column] * num_columns)
        a_grads = broadcast * grad_output
        # d(a/b) / dx for each x in a
        squared_inv = -a * b_column ** -2
        multiply_upstream = squared_inv * grad_output
        # sum along paths to get d(a/b) / db
        b_local_grads = np.sum(multiply_upstream, axis=1)
        return a_grads, b_local_grads


@tensor_op
class exp(Operation):
    """ e^x element-wise """

    @staticmethod
    def forward(ctx, a):
        value = np.exp(a)
        ctx.append(value)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        value = ctx[-1]
        return [value * grad_output]


def softmax_rows(logits: Tensor) -> Tensor:
    """Compute softmax of a batch of inputs.
    e^x / sum(e^x), where the sum is taken per-row

    Args:
        logits: [batch_size, num_classes] containing logits

    Returns:
        row-wise softmax of logits, i.e. each row will be a probability distribution.
    """
    exps = exp(logits)
    row_sums = sum_along_columns(exps)
    return divide(exps, row_sums)


def cross_entropy_loss(probabilities: Tensor, labels: Tensor) -> Tensor:
    """Compute mean cross entropy.

    Args:
        probabilities: [batch_size, num_labels], each row is a probability distribution
        labels: [batch_size, num_labels], each row is a probability distribution
            (typically, each row will be a one-hot)

    Returns:
        - 1 / batch_size * sum_i cross_entropy(prob[i] , labels[i])
    """
    logprobs = log(probabilities)
    cross_entropies = sum_along_columns(multiply(logprobs, labels))
    mean = reduce_mean(cross_entropies)
    negative_one = Tensor(np.array(-1.0))
    return multiply(negative_one, mean)