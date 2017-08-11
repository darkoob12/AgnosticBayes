
def ensemble_score(feat_sets, probs, p, n_samp):
    """
    aggregates all the given feature selection methods
    :param feat_sets: dictionary of feature selections
    :param probs: posterior probability of these sets
    :param p: total number of features
    :param n_samp: number of saples used to compute the posterior prab.
    :return: probability of each feature being in the optimal feature set
    """
    acc_sampling = np.zeros((p, 1))
    priors = get_priors(feat_sets, p)
    for s in range(n_samp):
        sample = np.random.permutation(p)  # a random sample of feature ranking
        posterior = np.zeros((p, 1))

        for x in range(p):
            D = get_feature_ranks(feat_sets, x)
            posterior[x] = exponential(D, sample[x]) * priors[x]


def get_feature_ranks(feat_sets, x):
    """
    INCOMPLETE and i don't remember what it's purpose
    :param feat_sets:
    :param x:
    :return:
    """
    r = np.zeros((len(feat_sets), 1))
    return r


def get_priors(feat_sets, p):
    """
    computes the prior probability for each feature empirically
    :param feat_sets: a dictionary of feature selections
    :param p: total number of features
    :return: an array of probs
    """
    ret = np.zeros((p, 1))
    for i in feat_sets:
        foo = feat_sets[i].scores_  # todo: not gonna work!
        s = {i: foo[i] for i in range(len(foo))}
        s = sorted(s.items(), key=lambda x: x[1], reverse=True)
        for i in range(p):
            ret[i] += s[i]

    for i in range(p):
        ret[i] /= len(feat_sets)
    return ret


def rank_to_lambda(r, d):
    """ (int, int) -> int
    computes the average difference in predicted rank and the given true rank
    :param r: true rank - it starts at 1
    :param d: number of features
    :return: average difference
    """
    ret = r * (r - 1) + (d - r) * (d - r + 1)
    ret /= 2 * d
    return ret



def analyze_feature_sets(feat_sets, n_feats):
    """
    statistically analyzes the obtained feature sets
    :param feat_sets: a list of trained feature selectors
    :return: None
    """
    counts = {}
    feats = np.zeros([len(feat_sets), n_feats])
    for i in feat_sets:
        s = feat_sets[i].get_support(True)
        feats[i, :] = s[:]
        for f in s:
            if f not in counts:
                counts[f] = 0
            counts[f] += 1
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for f in sorted_counts:
        print('feat #{0}\t\t{1}times'.format(f[0], f[1]))
    print(feats)



def comp_std(data, b = 200):
    """
    computing std of ranks of each feature
    :param data: whole data
    :param b: number of bootstraps
    """
    L, feat_sets = create_loss_matrix(b, data)
    F = np.zeros((data.p, b))

    for j in range(b):
        tmp = {k:feat_sets[j].scores_[k] for k in range(data.p)}
        tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
        for r in range(len(tmp)):
            F[tmp[r],j] = r+1
