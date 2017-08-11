import numpy as np

def argmin_with_ties(eLoss_kN):
    """
    using a set of risk vectors finds the best h for each one
    using simple min in each column
    :param eLoss_kN: k samples and N learning algorithms
    :return:
    """
    min_eLoss_k1 = np.min(eLoss_kN, 1).reshape(-1, 1)  # the min value for the k different bootstrap
    amin_kN = eLoss_kN == min_eLoss_k1  # for each bootstrap, flag each hp having the minimal eLoss (may be not unique)
    count_k1 = amin_kN.sum(1).reshape(-1, 1).astype(
        np.float)  # for each bootstrap, count how many were having the min (usually 1)

    return amin_kN / count_k1  # normalize the flag matrix so that sums to one for each bootstrap


def aB_prob(w_km, loss_mN):
    """
    posterior probability learned using the agnostic bayes method
    using the Efron's bootstrap
    :param w_km: k bootstraps of the m losses ???
    :param loss_mN: m samples and N learning algorithms
    :return: array of probabilities for each learning algorithm N * 1
    """

    eLoss_kN = np.dot(w_km, loss_mN)  # computational bottleneck O(kmN)

    amin_kN = argmin_with_ties(eLoss_kN)  # takes into account the possibility of having several mins
    prob_N = np.mean(amin_kN, 0)  # sum across the bootstrap

    np.testing.assert_almost_equal(prob_N.sum(), 1, )

    return prob_N


def build_bootstrap_matrix(k_bootstrap, m, essr, rng=None):
    """
    creates a matrix of bootstrap samples generated randomly
    :param k_bootstrap:  number of points in each bootstrap
    :param m: number of samples or parameters!!!!?????
    :param essr: effective sample size rate
    :param rng: random number generator seed
    :return:
    """
    bootstrap_matrix_km = np.zeros((k_bootstrap, m))
    for i in range(k_bootstrap):
        bootstrap_matrix_km[i, :] = bootstrap_weight(m, essr, rng)
    return bootstrap_matrix_km


def bootstrap_weight(n, essr=1.0, rng=None):
    """
    generate bootstrap samples from n points with specified effective
    sample size using the given random state and return sample weights
    :param n: number of validation samples
    :param essr: effective sample size rate
    :param rng: random number generator
    :return: ????
    """
    if rng is None: rng = np.random.RandomState()
    mSample = max(1, int(round(n * essr)))  # effective number of samples
    count_n = np.bincount(rng.randint(0, n, mSample), minlength=n)
    assert count_n.sum() == mSample
    prob_n = count_n.astype(np.float) / np.sum(count_n)
    return prob_n


def get_pp(loss, k):
    """
    computes the posterior probability for each of the sets using
    :param loss: loss matrix
    :param k: number of samples for bootstrap
    :return: a vector of probabilities
    """
    bootstrap = build_bootstrap_matrix(k, loss.shape[0], 1)
    prob = aB_prob(bootstrap, loss)
    return prob
