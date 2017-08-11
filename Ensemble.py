__author__ = 'Darkoob12'

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from itertools import combinations, chain
import random
import os
import warnings
import csv
import logging

from Dataset import *
from AgnosticBayes import *
from stability import *


class Constant:
    data_collection = [['colon_tumor', 2000],  # [name, number of features]
                       ['Lung_Cancer_michigan', 7129],
                       ['Leukemia2', 11225],
                       ['11_Tumors', 12533],
                       ['cns', 7129],
                       ['Prostate_Tumor', 10509],
                       ['ovarian_cancer_ms', 15154],
                       ['lung_cancer', 12600]]

    algorithms = ["no_fs", "single", "e_freq", "e_ab", "e_ab_c", "e_acc"]
    alg_indx = {"single": 0, "e_freq": 1, "e_ab": 2, "e_ab_c": 3}
    num_features = [3, 5, 7, 10, 15, 20, 25, 30]
    b = 300  # number of bootstrap subsets of data
    bs_sample_count = 4000  # number of samples for bootstrap sampling
    comb_s = (2, 3)  # size of combinations to consider
    runs = 10
    folds = 10


class CommonVars:
    L = None
    feat_sets = None


class Classifier:
    """
    divides data into appropriate validation/training sets and trains a classifier for each of
    the selected feature selection algorithms

    """

    def __init__(self, data_set, test_percent=0.1):
        """
        divides the data set into train/test partitions
        :param data_set: the data set
        :return: an object
        """
        self.data = data_set
        self.tr, self.ts = train_test_split(range(data_set.n), test_size=test_percent)
        self.losses = {}

    def evaluate(self, feat_sel):
        """ (object) -> num
        trains a classifier and test it on the test data using the selected
        features in the given feature selection.

        :param feat_set: a learned feature selection algorithm
        :return: returns a dict containing losses sample_index -> loss
        """
        xx = feat_sel.transform(self.data.x)
        clf = linear_model.LogisticRegression()
        clf.fit(xx[self.tr, :], self.data.y[self.tr])
        yp = clf.predict(xx[self.ts, :])
        test_num = len(self.ts)
        tmp = {}
        for i in range(test_num):
            tmp[self.ts[i]] = self.loss(self.data.y[self.ts[i]], yp[i])
        # scr = clf.score(xx[self.ts,:],self.data.y[self.ts])
        return tmp

    @staticmethod
    def loss(y, yp):
        """
        loss function defined for each pair of labels
        :param y: true label
        :param yp: predicted label
        :return: incurred loss
        """
        return 1 if y == yp else 0


def select_feature(data, n_feats):
    """ (matrix, int) -> object
    trains a feature selection algorithm and returns it.
    :param data: a data set (usually a bootstrap of the data)
    :return: trained feature selection algorithm
    """
    anova_filter = SelectKBest(f_regression, k=n_feats)
    anova_filter.fit(data.x, data.y)
    return anova_filter


def create_loss_matrix(bs_count, data, n_feats):
    """ (int, DataSet, int) -> np.matrix
    generates a number of bootstrap data and selects a feature set accordingly
    loss value of each set is computed using a fixed and deterministic classifier

    :param bs_count: number of bootstraps
    :param data: a loaded data set
    :return: a loss matrix  L and their corresponding feature selection objects
    """
    clf = Classifier(data)

    samples_num = len(clf.ts)
    L = np.empty((samples_num, bs_count))
    feature_sets = {}
    for i in range(bs_count):
        fs = select_feature(data.bootstrap(data.n), n_feats)
        losses = clf.evaluate(fs)
        L[:, i] = np.array(list(losses.values()))
        feature_sets[i] = fs

    return L, feature_sets


def obtain_feature_prob(features, probs, p, r_set):
    """ (dic, list, list)
    compute probability for each of the features and then we can rank them
    :param features: a dictionary containing feature sets
    :param probs: posterior probability for each of the sets
    :param p: total number of features in the dataset.
    :return: probability of selecting each feature
    """
    ft_cb = {}

    acc = np.zeros((p,), dtype=np.float)  # accumulated scores
    # counts = np.zeros((p,))  # number of occurrences for each feature
    for i in features:  # for each feature set
        ft_set = features[i].get_support(True)
        # 1 - single features
        for j in ft_set:
            acc[j] += probs[i]
            # counts[j] += 1  # todo: it's not used at all
        # 2 - feature combinations
        for r in r_set:
            cb_set = list(combinations(ft_set, r))
            for tup in cb_set:
                if tup in ft_cb:
                    ft_cb[tup] += probs[i]
                else:
                    ft_cb[tup] = probs[i]

    # now select features acording only to comb scores
    feat_score = {}
    total_score = acc.sum()
    for i in range(len(acc)):
        feat_score[i] = acc[i] / total_score
    return feat_score, ft_cb


def select_features_comb(feats, posteriors, p, r_set, num_feats):
    """
    selects features according to their computed probabilities and their combinatinos
    and their probabilities
    :param feats: a dictionary containing selected feature sets
    :param posteriors: posterior probability for each feature set
    :param p: total number of features in the dataset
    :param r_set: number of combinations to consider
    :param num_feats: number of features to select
    :return : final feature set
    """
    features_score, cb_score = obtain_feature_prob(feats, posteriors, p, r_set)
    # combine all scores
    overall_scores = cb_score
    for i in range(len(features_score)):
        overall_scores[(i,)] = features_score[i]

    # sort the combinations
    sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    selected = set()
    for i in range(len(sorted_scores)):
        if len(selected) >= num_feats:
            break
        for j in sorted_scores[i][0]:
            if len(selected) < num_feats:
                selected.add(j)
            else:
                break

    return [(x, features_score) for x in selected]


def proposed_fs(data, num_feats):
    """
    runs our algorithm and returns a set of selected features
    :param data: data set
    :param num_feats: number of features to select
    :return: a set of indices of selected features
    """
    L, feat_sets = create_loss_matrix(400, data, num_feats)
    # analyze_feature_sets(feat_sets)
    posterior_probs = get_pp(L, 10000)
    selected_features = select_features_comb(feat_sets, posterior_probs, data.p, (2, 3), num_feats)
    return selected_features


def freq_aggregate_fs(fs_objects, num_feats):
    """
    aggregates feature selections by adding the ranks for each feature
    after that selects features with lowset accumulated rank
    :param fs_objects: trined feature selection objects
    :param num_feats: number of features to select
    :return: ensembe feature set
    """
    # compute the selection frequency of each feature
    freq = {}
    for i in fs_objects:
        feats = fs_objects[i].get_support(True)
        for feat in feats:
            if feat not in freq:
                freq[feat] = 1
            else:
                freq[feat] += 1
    # sort features by their frequency
    sorted_feats = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_feats[0:num_feats]


def gen_file_name(path, m_name, r, f, p):
    """ (str, int, int, int) -> str
    gnerates a proper name for the file to be used for saving the corresponding feature set
    : param m_name: name of the methods being used
    : param r: run number
    : param f: fold number
    : param p: percentage of selected features
    : return : the file name
    """
    ret = 'results/' + path + '/' + m_name + str(r) + '_' + str(f) + '_' + str(p) + '.csv'
    return ret


def init_res_dic(runs, folds):
    """
    creates an empty data structure for storing the results
    :param runs: number of runs
    :param folds: number of cv folds
    :return: a dictionary object
    """
    res = dict()
    # [acc, f1m, auc] <- each member of dictionaries
    for a in Constant.algorithms:
        if a == 'no_fs':
            res[(a, 0)] = np.zeros((runs * folds, 3))
        else:
            for n in Constant.num_features:
                res[(a, n,)] = np.zeros((runs * folds, 3))
    return res


def acc_aggregate_fs(loss, feat_sets, num_feats, dim):
    """
    features are selected according to their average accuracy
    :param loss: prediction on the left-out set of samples
    :param feat_sets: selected features in difference runs
    :param num_feats: number of desired features
    :param dim: total number of features
    :return: final selected features
    """
    acc = np.sum(loss, axis=0) / loss.shape[0]
    scores = {}
    counts = np.array([0] * dim)
    for fs_ind in range(len(feat_sets)):
        for f in feat_sets[fs_ind].get_support(True):
            if f not in scores:
                scores[f] = 0
            scores[f] += acc[fs_ind]
            counts[f] += 1

    bs = len(feat_sets)
    for i in range(dim):
        if counts[i] != 0:
            scores[i] = scores[i] / counts[i]
            scores[i] = (scores[i] + (counts[i] / bs)) / 2

    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_feats[0:num_feats]


def alg_switch(alg_name, data, tr, ts, num_feats):
    """
    runs appropriate feature selection function according to the given name
    :param alg_name: name of the algorithm
    :param data: dataset object
    :param tr: index for train data
    :param ts: index for test data
    :param num_feats: number of desired features
    :return: (acc,f_measure,auc),features
    """
    feats = []
    ret = [0, 0, 0]
    if alg_name == 'no_fs':
        clf = KNeighborsClassifier()
        clf.fit(data.x[tr,], data.y[tr])
        y_predicted = clf.predict(data.x[ts,])
        ret[0] = accuracy_score(y_predicted, data.y[ts])
        ret[1] = f1_score(y_predicted, data.y[ts], average='weighted')
        if len(np.unique(data.y[ts])) == 2:
            try:
                ret[2] = roc_auc_score(data.y[ts], y_predicted)
            except:
                print('error in auc')
    elif alg_name == 'single':
        fs = SelectKBest(f_regression, num_feats)
        fs.fit(data.x[tr,], data.y[tr])
        clf = KNeighborsClassifier()
        clf.fit(fs.transform(data.x[tr,]), data.y[tr])
        y_predicted = clf.predict(fs.transform(data.x[ts,]))
        ret[0] = accuracy_score(y_predicted, data.y[ts])
        ret[1] = f1_score(y_predicted, data.y[ts])
        if len(np.unique(data.y[ts])) == 2:
            try:
                ret[2] = roc_auc_score(data.y[ts], y_predicted)
            except:
                print('error in auc')
        feats = fs.get_support(True)
    elif alg_name == 'e_ab_c' or 'e_ab' or 'e_freq' or 'e_acc':
        # creates bootstrap datas and select features accordingly!
        if CommonVars.L is None:
            CommonVars.L, CommonVars.feat_sets = create_loss_matrix(Constant.b, data, num_feats)

        # compute probs for features
        if alg_name == 'e_ab':
            posterior_probs = get_pp(CommonVars.L, Constant.bs_sample_count)
            selected_features = select_features_comb(CommonVars.feat_sets, posterior_probs, data.p, (), num_feats)
        elif alg_name == 'e_ab_c':
            posterior_probs = get_pp(CommonVars.L, Constant.bs_sample_count)
            selected_features = select_features_comb(CommonVars.feat_sets, posterior_probs, data.p, Constant.comb_s,
                                                     num_feats)
        elif alg_name == 'e_freq':
            selected_features = freq_aggregate_fs(CommonVars.feat_sets, num_feats)
        else:
            selected_features = acc_aggregate_fs(CommonVars.L, CommonVars.feat_sets, num_feats, data.p)

        s1 = [x[0] for x in selected_features]

        clf = KNeighborsClassifier()
        clf.fit(data.x[np.ix_(tr, s1)], data.y[tr])
        y_predicted = clf.predict(data.x[np.ix_(ts, s1)])
        ret[0] = accuracy_score(y_predicted, data.y[ts])
        ret[1] = f1_score(y_predicted, data.y[ts])
        if len(np.unique(data.y[ts])) == 2:
            try:
                ret[2] = roc_auc_score(data.y[ts], y_predicted)
            except:
                print('error in auc')
        feats = s1
    else:
        pass
        # error handling!!

    return ret, feats


def save_avg(res, data_name):
    """
    compute average and std of multiple runs and save to a file
    :param res: a dictionary containing the data
    :param data_name: name of dataset for file name
    :return: none
    """
    out = [['algorithm', '# features', 'accuracy_avg', 'accuracy_std', 'fmeasure_avg', 'fmeasure_std', 'auc_avg',
            'auc_std']]
    for key in res:
        record = [key[0], key[1]]
        runs = res[key]
        avg = np.mean(runs, 0)
        std = np.std(runs, 0)
        for i in range(len(avg)):
            record.append(avg[i])
            record.append(std[i])
        out.append(record)
    with open(data_name + '_res' + '.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(out)
        f.flush()


def experiment(data, runs, folds, data_name):
    """
    performs the experiment and compares the results of different methods
    we will use the same classifier for both of the feature selection methods
    :param data_name: name of the data set used for output
    :param data: a dataset object containing the training data
    :param runs: number of independent runs of each algorithm
    :param folds: number of folds in a cross-validation for each run
    :return: None
    """
    # LOG: create folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join('results', data_name)
    os.makedirs(base_path, exist_ok=True)

    # save accuracies
    res = init_res_dic(runs, folds)
    obtained_features = {}
    for nf in Constant.num_features:
        obtained_features[nf] = np.zeros((4, runs * folds, nf))
    stab = {}

    rf = 0
    for r in range(runs):
        print('********* run = {0} *********\n'.format(r))
        cross_val = KFold(data.n, folds, shuffle=True, random_state=13)
        k = 0  # i should count manually! number of folds
        for tr, ts in cross_val:
            print('fold = {0}:'.format(k))
            alg = 'no_fs'
            nf = 0
            res[(alg, nf)][rf, :], a = alg_switch(alg, data, tr, ts, nf)
            print('{0:s} : '.format(alg), end='')
            print('({0:1.3f},{1:1.3f},{2:1.3f})'.format(res[(alg, nf)][rf, 0],
                                                        res[(alg, nf)][rf, 1],
                                                        res[(alg, nf)][rf, 2]), end='')
            for nf in Constant.num_features:
                print('\n====> desired number of features : {0} \n'.format(nf))
                for alg in Constant.algorithms:
                    if alg == 'no_fs':
                        continue
                    print('{0:s} : '.format(alg), end='')
                    res[(alg, nf,)][rf, :], obtained_features[nf][Constant.alg_indx[alg], rf, :] = alg_switch(
                        alg, data, tr, ts, nf)
                    print('({0:1.3f},{1:1.3f},{2:1.3f}) | '.format(res[(alg, nf,)][rf, 0],
                                                                   res[(alg, nf,)][rf, 1],
                                                                   res[(alg, nf,)][rf, 2]), end='')
                # reset common variables
                CommonVars.L = None
                CommonVars.feat_sets = None

            # next fold
            rf += 1
            k += 1
            print()  # next fold

    # save the results for current number of features
    for nf in Constant.num_features:
        for key in Constant.alg_indx:
            out_name = "feats_" + key + "_" + str(nf) + ".csv"
            tmp = obtained_features[nf][Constant.alg_indx[key], :, :]  # a matrix feats * runs
            tmp = np.transpose(tmp)
            np.savetxt(os.path.join(base_path, out_name), tmp, fmt="%d", delimiter=',')
            # compute the stability measures
            stab[(key, nf)] = [total_stability(tmp, data.p, 'jaccard'), total_stability(tmp, data.p, 'kuncheva')]

    # save the final results
    for key in res:
        out_name = key[0] + '_' + str(key[1]) + '.csv'
        np.savetxt(os.path.join(base_path, out_name), res[key], fmt='%1.11f', delimiter=',')
    # compute average and standard deviation
    save_avg(res, data.name)

    with open(os.path.join(base_path, 'stability.csv'), 'w', newline='') as f:
        f.write('method,num_feats,jaccard,kuncheva\n')
        for key in stab:
            f.write('{0:s},{1:d},{2:1.11f},{3:1.11f}\n'.format(key[0], key[1], stab[key][0], stab[key][1]))
        f.flush()
    # print something to screen that says the experiment is over
    print('\n\n=========================FINISHED====================================')


def single_run_test(data, nf):
    """ (object, int) -> None
    run multiple algorithms by dividing the data to test/train parts
    :param data: data set object
    :param nf: number of features to select
    """
    alg_list = ["single", "e_freq", "e_ab", "e_ab_c", "e_acc"]
    tr, ts = train_test_split(range(data.n))
    alg = 'no_fs'
    res, a = alg_switch(alg, data, tr, ts, 0)
    print('{0:s}\t:\t'.format(alg), end='')
    print('({0:1.3f},{1:1.3f},{2:1.3f})'.format(res[0], res[1], res[2]))

    for alg in alg_list:
        print('{0:s}\t:\t'.format(alg), end='')
        res, a = alg_switch(alg, data, tr, ts, nf)
        print('({0:1.3f},{1:1.3f},{2:1.3f})'.format(res[0], res[1], res[2]))
    # reset common variables
    CommonVars.L = None
    CommonVars.feat_sets = None


def multiple_run(data, r, n_feats):
    """
    runs the algorithm for a given number of times and reports the average results
    :param data: dataset object
    :param r: number of independent runs
    :param n_feats: number of features to select
    """
    s = np.zeros((r, n_feats))
    for i in range(r):
        tr, ts = train_test_split(range(data.n))
        fs = SelectKBest(f_regression, n_feats)
        pipe = Pipeline([('feat_selector', fs), ('knn', KNeighborsClassifier())])
        pipe.fit(data.x[tr,], data.y[tr])
        tmp_scr = pipe.predict_proba(data.x[ts,])
        s1 = fs.get_support(indices=True)

        # foo = proposed_fs(data, 20)  # selected features by our method and their probabilities
        # s1 = [x[0] for x in foo]
        # clf = KNeighborsClassifier()
        # clf.fit(data.x[np.ix_(tr, s1)], data.y[tr])
        # # print(clf.score(data.x[np.ix_(ts, s1)], data.y[ts]))
        # tmp_scr = clf.predict_proba(data.x[np.ix_(ts, s1)])
        # print(roc_auc_score(data.y[ts], tmp_scr[:, 1]))
        ## save features
        s1.sort()
        s[i,] = s1

    np.savetxt('simple_feats.csv', s)
    print(s)
    return s


def get_data_set_path(name):
    """
    simple function that create relative path for a dataset
    """
    base = 'datasets/'
    suffix = '.csv'
    return base + name + suffix


def print_line(c=80, ch='-'):
    for i in range(c):
        print(ch, end='')
    print()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    data_name = 'madelon'
    data = DataSet()
    data.read_data(get_data_set_path(data_name), data_name)
    data.normalize_z()
    print_line(ch='_')
    print('\n data name = {0:s}\t samples={1:d}\tfeatures={2:d}'.format(data.name, data.n, data.p))
    print_line()
    single_run_test(data, 40)

if __name__ == "__main__1":
    print(__file__)
    # ignore all the warning due to the compatibility issues
    warnings.filterwarnings('ignore')

    for d in Constant.data_collection:
        name = d[0]
        # read the data
        data = DataSet()
        data.read_data(get_data_set_path(name), name)
        data.normalize_z()
        print('\n data name = {0:s}\t samples={1:d}\tfeatures={2:d}'.format(name, data.n, data.p))
        experiment(data, Constant.runs, Constant.folds, name)
