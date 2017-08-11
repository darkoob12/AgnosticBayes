from Dataset import *
from Ensemble import *
from stability import *
import numpy as np

from sklearn.metrics import f1_score


def gen_file_name(alg_name, data_name, feat_p):
    ret = 'results/'
    ret += data_name + '/'
    ret += 'acc_'
    ret += alg_name + '_'
    ret += str(feat_p) + '.csv'
    return ret


def gen_file_name_feats(alg_name, data_name, run, fold, feat_p):
    ret = 'results/'
    ret += data_name + '/'
    ret += alg_name + '_'
    ret += str(run) + '_' + str(fold) + '_' + str(feat_p) + '.csv'
    return ret


def save_dict(file, dic):
    """
    stores a give dictionary to a csv file
    :param file: file name
    :param dic: dictionary - which has to be in a specific format
    :return: none
    """
    import csv
    with open(file, 'w', newline='') as f:
        wr = csv.writer(f, delimiter=',')
        for data in dic:
            for alg in dic[data]:
                for p in range(5):
                    foo = [data, alg, str(p + 1), str(dic[data][alg][p, 0]), str(dic[data][alg][p, 1])]
                    print(foo)
                    wr.writerow(foo)


def integrate_files():
    """
    combine the selected features of several runs in a single file
    :return: none
    """
    from scipy import stats
    # the same loops in the experiment
    percents = [3, 5, 7, 10, 15, 20]
    for d_ in Constant.data_collection:
        dn = d_[0]
        for p in percents:
            print('___________________')
            all_nums = np.zeros((100, 4))
            k = 0
            for alg in Constant.algorithms:
                file_name = gen_file_name(alg, dn, p)
                acc = np.loadtxt(file_name, delimiter=',')
                all_nums[:, k] = acc.reshape(100)
                k += 1
            np.savetxt(dn+'_'+str(p)+'.csv',all_nums,delimiter=',')
            #h, pv = stats.friedmanchisquare(all_nums[:, 0], all_nums[:, 1], all_nums[:, 2], all_nums[:, 3])
            #if pv <= 0.1:
            #    print("{0}_{1} => {2:.6f}".format(data, p, pv))


def compute_avg_std():
    """
    load the results and compute the average and standard deviation of various runs
    :return: a dictionary containing the computed values
    """

    # the same loops in the experiment
    data_names = ['11_Tumors', 'Leukemia2', 'pcmac', 'Prostate_Tumor', 'warpPI']
    algorithms = ["single", "e_freq", "e_ab", "e_ab_c"]
    percents = (1, 2, 3, 4, 5)

    stats = {}

    for data in data_names:
        stats[data] = {}
        for alg in algorithms:
            stats[data][alg] = np.zeros((5, 2))
            for p in percents:
                file_name = gen_file_name(alg, data, p)
                acc = np.loadtxt(file_name, delimiter=',')
                stats[data][alg][p - 1, 0] = np.mean(acc)
                stats[data][alg][p - 1, 1] = np.std(acc)

    return stats


def run_again():
    """
    loads the selected features for each algorithm and computes F1 measure
    after running the knn algorithm
    """
    data_names = ['11_Tumors', ]  # ,'Leukemia2','pcmac','Prostate_Tumor','warpPI']
    algorithms = ["sinle", "e_freq", "e_ab", "e_ab_c"]
    percents = (1, 2, 3, 4, 5)

    f_scores = np.zeros((100, 20))
    k = 0
    for dn in data_names:
        data = DataSet()
        data.read_data(get_data_set_path(dn))
        data.normalize()
        for alg in algorithms:
            for p in percents:
                tmp = []
                for r in range(10):
                    cross_val = KFold(data.n, 10, shuffle=True)
                    fold = 0
                    for tr, ts in cross_val:
                        # read the file
                        f_name = gen_file_name_feats(alg, dn, r, fold, p)
                        if alg in ('sinle', 'e_ab'):
                            feats = np.loadtxt(f_name, delimiter=',', dtype=int)
                        else:
                            feats = np.loadtxt(f_name, delimiter=',')
                            a = []
                            for row in feats:
                                a.append(int(row[0]))
                            feats = np.array(a)
                        print(f_name)
                        fold += 1
                        clf = KNeighborsClassifier()
                        clf.fit(data.x[np.ix_(tr, feats)], data.y[tr])
                        f1 = f1_score(data.y[ts], clf.predict(data.x[np.ix_(ts, feats)]), average='weighted')
                        tmp.append(f1)
                f_scores[:, k] = np.array(tmp)
                k += 1
                # loop on the percentages


if __name__ == "__main__":
    integrate_files()

if __name__ == "__main__222":
    algorithms = ["e_freq", "e_ab", "e_ab_c"]
    dnf = [3, 5, 7, 10, 15, 20]  # desired number of features

    stabs = {}
    for d_ in Constant.data_collection:
        dn = d_[0]
        n = d_[1]
        for alg in algorithms:
            for p in dnf:
                # ! a dumb move
                tmp_acc = []
                for r in range(10):  # each run
                    for f in range(10):  # each fold
                        # read the file
                        f_name = gen_file_name_feats(alg, dn, r, f, p)
                        if alg in ('sinle', 'e_ab'):
                            feats = np.loadtxt(f_name, delimiter=',', dtype=int)
                        else:
                            feats = np.loadtxt(f_name, delimiter=',')
                            a = []
                            for row in feats:
                                a.append(int(row[0]))
                            feats = np.array(a)
                        tmp_acc.append(feats)
                # the matrix contain selected features for 100 runs alg_data_p
                # convert to ndarray
                feats_matrix = np.matrix(tmp_acc)
                np.savetxt(dn + alg + str(p) + '.csv', feats_matrix, delimiter=',')
                # s_total = total_stability(feats_matrix, n)
                # print(dn + alg + str(p) + ' => ' + str(s_total))
                # stabs[(dn, alg, p)] = s_total
                # loop on the percentages
