from itertools import combinations

import numpy as np
import random


class DataSet:
    """
    represents a data set and required methods for editing them
    : x: features matrix
    : y: labels vector
    : n: number of samples
    : p: number of features
    """

    def __init__(self, n=-1, p=-1, dn='no_name'):
        """
        creates an empty data set
        :param n: number of samples (-1)
        :param p: number of features (-1)
        :return: an empty data set
        """
        self.name = dn
        if n < 0 or p < 0:
            self.x = []
            self.y = []
            self.n = -1
            self.p = -1
        else:
            self.n = n
            self.x = np.empty((n, p))
            self.y = np.empty((n, 1))
            self.p = p

    def copy(self, ind):
        """
        creates a new DataSet object containing a subset of the current one
        :param ind: indices to be included
        :return: a new DataSet
        """
        ret = DataSet(len(ind), self.p)
        ret.x = self.x[ind,]
        ret.y = self.y[ind,]
        return ret

    def read_data(self, filename, dn=''):
        """
        reads data from file and convert them to numbers and save them into matrix
        it is assumed that the first value in each record is class label
        :param dn: dataset name used for reporting
        :param filename: name of the data file
        :return: none
        """
        if dn != '':
            self.name = dn
        # read the data set
        f = open(filename)
        x = []
        y = []
        for line in f:
            row = line.split(',')
            y.append(int(row[0]))  # adding the label
            tmp = []
            for i in range(1, len(row)):
                tmp.append(float(row[i]))
            x.append(tmp)
        f.close()
        self.x = np.matrix(x)
        self.y = np.array(y)
        self.n = len(y)
        self.p = self.x.shape[1]  # number of columns of the data matrix

    def normalize(self):
        """
        normalize the feature vectors using empirical max and min of features
        :return: none
        """
        m_ = np.amin(self.x, 0)
        M_ = np.amax(self.x, 0)

        self.x = (M_ - self.x) / (M_ - m_)
        # for i in range(self.p):
        #     self.x[:,i] = (M_[0,i] - self.x[:,i]) / (M_[0,i] - m_[0,i])

    def normalize_z(self):
        """
        normalize using z-score    
        """
        sig_ = np.std(self.x, 0)
        mu_ = np.mean(self.x, 0)

        self.x = (self.x - mu_) / sig_

    def add(self, features, label, index):
        """
        adds a new sample
        :param features: a feature vector
        :param label: label of the vector
        :param index: at which point the new vector should be added
        """
        self.x[index, :] = features[:]
        self.y[index] = label

    def bootstrap(self, size):
        """ (matrix) -> matrix
        selects randomly with replacement from the given data set

        :param data: a source data set
        :param size: number of data to select
        :return: a selected bootstrap of data
        """
        bs = DataSet(size, self.p)
        for i in range(size):
            ind = random.choice(range(self.n))
            bs.add(self.x[ind, :], self.y[ind], i)

        return bs

    @staticmethod
    def feat_comb(p, r_set):
        ret = {}
        for r in r_set:
            combs = list(combinations(range(p - 1), r))
            for tup in combs:
                ret[tup] = 0
        return ret
