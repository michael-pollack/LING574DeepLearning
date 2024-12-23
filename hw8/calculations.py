import numpy as np
queries = np.ndarray([
    [0.2, 0.4],
    [0.8, 0.6],
    [1.0, 1.2]
])

keys = np.ndarray([
    [0.1, 0.7, 0.9, 1.5],
    [0.3, 0.5, 1.1, 1.3]
])

mask = np.ndarray([
    [0, 0, 0, 0],
    [0, 0, 0, -np.inf],
    [0, 0, -np.inf, -np.inf]
])

compatibility = np.multiply(keys, queries)
print("COMPATIBILITY: ")
print(compatibility)

masked_compatibility = compatibility + mask
print("MASKED COMPATIBILITY: ")
print(masked_compatibility)

def softmax(x):
    return(np.exp(x)/np.exp(x).sum)

attention = softmax(masked_compatibility)
print("ATTENTION: ")
print(attention)

output = np.multiply(attention, values)
print("OUTPUT: ")
print(output)

