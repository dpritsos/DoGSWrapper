

import numpy as np


def cosine_similarity(v1, v2):

    # Convert Arrays and Matricies to Matrix Type for preventing unexpeted
    # error, such as returning vector instead of scalar
    v1 = np.matrix(v1)
    v2 = np.matrix(v2)

    l = lambda v1, v2: (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0 or (np.linalg.norm(v1) * np.linalg.norm(v2))

    # print l(v1,v2)

    return v1 * np.transpose(v2) / l(v1, v2)


def minmax_similarity(v1, v2):

    return np.sum(np.min(np.vstack((v1, v2)), axis=0)) / np.sum(np.max(np.vstack((v1, v2)), axis=0))


v1 = np.array([1, .1, 1, 1, 1])
v2 = np.array([.1, .1, 1, 0.1, 0.1])

print cosine_similarity(v1, v2)
print minmax_similarity(v1, v2)
