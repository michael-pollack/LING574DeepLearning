import numpy as np

from edugrad.ops import Operation, tensor_op


@tensor_op
class sigmoid(Operation):
    @staticmethod
    def forward(ctx, a):
        sigmoid = 1 / (1 + np.exp(-a))
        ctx.append(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid, = ctx
        derivative = sigmoid * (1 - sigmoid)
        return derivative * grad_output


@tensor_op
class log(Operation):
    @staticmethod
    def forward(ctx, a):
        ctx.append(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx
        derivative = (1 / a)
        return np.multiply(derivative, grad_output)


@tensor_op
class multiply(Operation):
    """Element-wise multiplication. """

    @staticmethod
    def forward(ctx, a, b):
        ctx.append(a)
        ctx.append(b)
        return np.multiply(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx
        grad_a = np.multiply(grad_output, b)
        grad_b = np.multiply(grad_output, a)
        return grad_a, grad_b


@tensor_op
class sum_along_columns(Operation):
    @staticmethod
    def forward(ctx, a):
        ctx.append(a)
        return np.sum(a, axis=1)

    @staticmethod
    def backward(ctx, grad_output):
        return [np.ones(ctx[-1].shape) * grad_output[:, np.newaxis]]


@tensor_op
class lookup_rows(Operation):
    """Given a matrix of size [m, n] and an array of integer indices
    [i0, i1, ..., in], return the relevant rows of the matrix.
    """

    @staticmethod
    def forward(ctx, matrix, indices):
        ctx.append(matrix.shape)
        ctx.append(indices)
        return matrix[indices]

    @staticmethod
    def backward(ctx, grad_output):
        shape, indices = ctx
        grads = np.zeros(shape)
        # this is some numpy magic: `indices` may have repeats of a given token index,
        # but if we just do grads[indices] += grad_output, it won't add up the rows
        # from grad_output for each occurance of the same index; this method accumulates
        # all of those sums, which is what's needed for the gradients
        np.add.at(grads, indices, grad_output)
        return [grads, np.zeros(indices.shape)]
