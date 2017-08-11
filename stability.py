import numpy as np


def kuncheva(A, B, n):
    """ (array,array,n) -> num
    computes the Kuncheva index for two given set of integer numbers.
    these sets are supposed to have the same number of elements
    """
    A = np.squeeze(np.asarray(A))
    B = np.squeeze(np.asarray(B))
    assert (len(A) == len(B))
    s = len(A)
    assert (s != 0)
    assert (s != n)
    intersect = np.intersect1d(A, B)
    r = len(intersect)
    ret = r * n - s * s
    ret /= s * n - s * s
    return ret


def total_stability(F, n, measure):
    """ (matrix, num) -> num
    the columns of matrix are the selected features in different runs
    n is the number of features
    this function returns the average mutual Kuchenva index of the runs for one
    algorithm
    :param n: number of features
    :param F: a matrix each row represents a feature set
    :param measure: name of similarity measure 'jaccard' or 'kuncheva'
    """
    assert (isinstance(F, np.ndarray))
    r, k = F.shape
    ret = 0
    for i in range(r):
        for j in range(i + 1, r):
            if measure == 'kuncheva':
                ret += kuncheva(F[i, :], F[j, :], n)
            elif measure == 'jaccard':
                ret += jaccard_index(F[i, :], F[j, :])
    ret /= k * (k - 1) / 2
    return ret


def jaccard_index(A, B):
    """
    similarity index
    :param A: one dimensional array
    :param B: one dimensional array
    :return: numeric indicator of similarity
    """
    A = np.squeeze(np.asarray(A))
    B = np.squeeze(np.asarray(B))
    if len(A) == 0 and len(B) == 0:
        return 1
    intersect = np.intersect1d(A, B)
    union = np.union1d(A, B)
    return len(intersect) / len(union)


def gen_random_featureset(k,n,r):
    ret = np.zeros((r,k))
    for i in range(r):
        ret[i,:] = np.random.choice(n,k)

    return ret


if __name__ == '__main__':
    print('hello, world!')

    ds = gen_random_featureset(14,200,100)

    foo = total_stability(ds,200,'jaccard')
    print(foo)

