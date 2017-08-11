# -*- coding: utf-8 -*-
'''
Created on 2012-12-17

@author: alexandre
'''

import numpy as np
from scipy.special import gammaln as gl, betaln as bl, multigammaln as mgl, psi
from scipy import optimize as opt
from scipy.linalg import eigvalsh
from scipy.optimize import fmin_l_bfgs_b
import random


verbosity = 1

"""
Main file to compute the Agnostic Bayes probability according to 3 methods
Use either LossSampler_bootstrap, LossSampler_t or LossSampler_Dirichlet to 
calculate these prob. They all extends HpProb, from which you can call getHpProb( loss_md ) to obtain the probability distribution.
"""

def normalizeLogProb(log_p_nd):
    """
    the probability distribution is along the d axis. 
    If one dimensionnal, will consider n = 1
    :param log_p_nd:
    :return:
    """

    reshape = log_p_nd.ndim == 1
    if reshape:
        log_p_nd = log_p_nd.reshape(1, -1)

    log_p_max_n1 = log_p_nd.max(1).reshape(-1, 1)

    p_nd = np.exp(log_p_nd - log_p_max_n1)  # pre-normalize for better numerical precision
    p_nd /= p_nd.sum(1).reshape(-1, 1)

    if reshape:
        p_nd = p_nd.reshape(-1)

    return p_nd


def softmin(x_nd, alpha=1e10):
    """
    ???
    :param x_nd:
    :param alpha:
    :return:
    """
    return normalizeLogProb(-x_nd * alpha)

class ProbBest:
    def __init__(self):
        self.count_d = None

    def updateFromLoss(self, expLoss_dk):
        '''

        :param expLoss_dk:
        :return:
        '''
        assert not np.any(np.isnan(expLoss_dk))
        assert np.all(np.isfinite(expLoss_dk))

        d, _k = expLoss_dk.shape
        if self.count_d is None:
            self.count_d = np.zeros(d)

        minExpLoss_dk = softmin(expLoss_dk.T, alpha=1e8).T
        #        print uHist( np.argmax(minExpLoss_dk,1) )
        #        print uHist( minExpLoss_dk)
        self.count_d += np.sum(minExpLoss_dk, 1)

    def getProb(self):
        assert np.sum(self.count_d) > 0
        return self.count_d / np.sum(self.count_d)


class ProbBest_oob:
    def __init__(self):
        self.count_dm = None
        self.count_d = None

    def updateFromLoss_oob(self, expLoss_dk, free_samples_km):

        # verification
        assert not np.any(np.isnan(expLoss_dk))
        assert np.any(np.isfinite(expLoss_dk))
        assert expLoss_dk.shape[1] == free_samples_km.shape[0]

        if self.count_d is None:  # init
            d = expLoss_dk.shape[0]
            m = free_samples_km.shape[1]
            self.count_d = np.zeros(d)
            self.count_dm = np.zeros((d, m))

        minExpLoss_1k = np.min(expLoss_dk, 0).reshape(1, -1)
        flag_dk = expLoss_dk == minExpLoss_1k
        count_1k = flag_dk.sum(0).reshape(1, -1).astype(np.float)
        assert count_1k.shape[1] == expLoss_dk.shape[1]

        if np.any(count_1k != 1):  # most of the time all count should be 1.
            flag_dk = flag_dk / count_1k

        self.count_d += np.sum(flag_dk, 1)
        self.count_dm += np.dot(flag_dk, free_samples_km)

    def getProb(self):
        assert np.sum(self.count_d) > 0
        return self.count_d / np.sum(self.count_d)

    def getProb_oob(self):
        assert all(np.sum(self.count_dm, 0) > 0)
        return self.count_dm / np.sum(self.count_dm, 0).reshape(1, -1)


# This function was copied from scikit-statsmodels
# written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
 
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
 
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable 
    '''

    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n) / df

    for epsilon in np.logspace(-20, -5, 10):  # just to avoid a nan bug
        z = np.random.multivariate_normal(np.zeros(d), S, (n,))
        if np.any(np.isnan(z)):
            print
            'WARNING : got nan. Adding %.3g on the diagonal' % epsilon
            S += np.identity(d) * epsilon
        else:
            break

    y_dk = m + z / np.sqrt(x)[:, None]

    if np.any(np.isnan(y_dk)):
        from graalUtil.file import writePkl

        writePkl((S, n), 'S.pklz')
        #         print unicodeHist(x, 'x')
        #        print unicodeHist( z, 'z' )
        print
        'nan in z %d/%d' % (np.sum(np.isnan(z)), np.prod(z.shape))

    return y_dk  # same output format as random.multivariate_normal


class BetaLogLike:
    def __init__(self, N, kL):
        self.N = N
        if np.all(kL <= 1):
            print
            'WARNING : k<=1 for all k. setting first one to 2.'
            kL = np.copy(kL).astype(np.float)
            kL[0] = 2
        self.kL = kL

    #        print self.kL


    def eval(self, beta):
        N = self.N
        n = len(self.kL)
        kSum = np.sum(self.kL)
        beta_N = beta / N  # often a really small number
        return gl(self.kL + beta_N).sum() + gl(beta) - n * gl(beta_N) - gl(beta + kSum)

    def comp(self, beta):
        N = self.N
        n = len(self.kL)
        kSum = np.sum(self.kL)
        beta_N = beta / N  # often a really small number
        #        print self.kL + beta_N
        print
        beta, beta_N

        return gl(self.kL + beta_N).sum(), n * gl(beta_N), gl(kSum), bl(beta, kSum)


    def toMinimize(self, lnBeta):
        beta = np.exp(lnBeta)
        return -self.eval(beta)

    def opt(self):
        optRes = opt.golden(self.toMinimize, brack=(-100, 100), full_output=True)
        return np.exp(optRes[0])

    def getAlphaStar(self):
        betaStar = self.opt()
        print
        'betaStar : %.3g' % betaStar
        return betaStar


class MML_t:
    def __init__(self, loss_dm, t):
        d = loss_dm.shape[0]

        _mu_pr_d, T_pr_dd, _kappa_pr, nu_pr = t.getPrior(d)
        _mu_d, T_dd, _kappa, nu = t.getPosterior(loss_dm)

        self.T_dd = T_dd - T_pr_dd  # just remove the prior to add it later
        self.nu = nu
        self.nu_pr = nu_pr

        print
        self.nu_pr, self.nu

    def eval(self, l):
        #        print l
        d = self.T_dd.shape[0]
        T_dd = self.T_dd + np.identity(d) * l
        #         print unicodeHist(T_dd, 'T_dd')
        sgn, lnDetT = np.linalg.slogdet(T_dd)
        assert sgn >= 0
        print
        lnDetT, d * np.log(l)
        print
        self.nu_pr, self.nu

        eigVal_d = eigvalsh(T_dd)
        lnDet = np.log(eigVal_d).sum()
        print
        'lnDet : ', lnDet
        print
        'eigval : ', np.round(eigVal_d, 3)

        return self.nu_pr * d * np.log(l) - self.nu * lnDetT

    def comp(self, l):
        d = self.T_dd.shape[0]
        T_dd = self.T_dd + np.identity(d) * l

        sgn, lnDetT = np.linalg.slogdet(T_dd)
        assert sgn >= 0

        return self.nu_pr * d * np.log(l), -self.nu * lnDetT


    def toMinimize(self, ln_l):
        l = np.exp(ln_l)
        return -self.eval(l)

    def opt(self):
        optRes = opt.golden(self.toMinimize, brack=(-3, 3), full_output=True)
        return np.exp(optRes[0])


def tPosterior(k0, nu0, mu0_d, T0_dd, avgLoss_d, stdLoss_dd, ess, m):
    if m == 0:
        return k0, nu0, mu0_d, T0_dd

    dMu_d = mu0_d - avgLoss_d

    k = k0 + ess
    nu = nu0 + ess
    mu_d = (k0 * mu0_d + ess * avgLoss_d) / k
    T_dd = T0_dd + stdLoss_dd * ess / m + k0 * ess * np.outer(dMu_d, dMu_d) / k

    return k, nu, mu_d, T_dd


def tSufficientStats(loss_dm):
    '''

    :param loss_dm:
    :return:
    '''
    d, m = loss_dm.shape

    avgLoss_d = np.mean(loss_dm, 1)
    assert avgLoss_d.shape == (d,)

    dLoss_dm = loss_dm - avgLoss_d.reshape(-1, 1)

    stdLoss_dd = np.dot(dLoss_dm, dLoss_dm.T)
    assert stdLoss_dd.shape == (d, d)

    return avgLoss_d, stdLoss_dd, m


class MML_tFull:
    def __init__(self, essr=1., k0=None, nu0=None, mu0=None, t0=None):

        self.essr = essr

        self.k0 = k0
        self.nu0 = nu0
        self.mu0 = mu0
        self.t0 = t0

        self.lastPoint = None
        self.lnDet = None

        self.verbose = 0

    def short_name(self):
        if (self.mu0 is None) or (self.t0 is None):
            return 'mml'
        else:
            return 'fix'

    def __str__(self):
        nameL = ['k0', 'nu0', 'mu0', 't0']
        strL = []
        for name in nameL:
            val = getattr(self, name)
            if val is None:
                strL.append('%s : mml' % name)
            else:
                strL.append('%s : %.3g' % (name, val))
        return ', '.join(strL)


    def convertVar(self, x):
        i = 0
        d = self.d

        if self.k0 is None:
            k0 = np.exp(x[i])
            i += 1
        else:
            k0 = self.k0

        if self.nu0 is None:
            nu0 = np.exp(x[i]) + d - 1
            i += 1
        else:
            nu0 = self.nu0

        if self.mu0 is None:
            mu0 = x[i]
            i += 1
        else:
            mu0 = self.mu0

        if self.t0 is None:
            t0 = np.exp(x[i])
            i += 1
        else:
            t0 = self.t0

        return k0, nu0, mu0, t0

    def invConvertVar(self, k0, nu0, mu0, t0):
        d = self.d
        #        print nu0, d, nu0- d +1, np.log(nu0-d+1)
        return np.array((np.log(k0), np.log(nu0 - d + 1), mu0, np.log(t0)))


    def logMl_(self, k0, nu0, mu0, t0):
        self.lastPoint = (k0, nu0, mu0, t0)
        d = self.avgLoss_d.shape[0]
        ess = self.essr * self.m
        mu0_d = mu0 * np.ones(d)
        T0_dd = np.identity(d) * t0
        k, nu, _mu_d, T_dd = tPosterior(k0, nu0, mu0_d, T0_dd, self.avgLoss_d, self.stdLoss_dd, ess, self.m)

        sgn, lnDet = np.linalg.slogdet(T_dd)
        self.lnDet = lnDet
        assert sgn >= 0

        d_2 = d / 2.

        return mgl(nu * .5, d), -mgl(nu0 * .5, d), np.log(t0) * nu0 * d_2, -nu * lnDet * .5, np.log(k0 / k) * d_2

    def logMl(self, *argL):
        return sum(self.logMl_(*argL))

    def dLogMl(self, k0, nu0, mu0, t0):
        d = self.avgLoss_d.shape[0]
        m = self.m

        mu0_d = mu0 * np.ones(d)
        T0_dd = np.identity(d) * t0
        k, nu, _mu_d, T_dd = tPosterior(k0, nu0, mu0_d, T0_dd, self.avgLoss_d, self.stdLoss_dd, m, m)

        dMu_d = mu0_d - self.avgLoss_d
        Tinv_dd = np.linalg.inv(T_dd)

        if self.k0 is None:
            G = np.outer(dMu_d, dMu_d)
            trTG = (Tinv_dd * G).sum()
            dk0 = (1. / k0 - 1. / k) * d * 0.5 - nu * 0.5 * (m / k) ** 2 * trTG
        else:
            dk0 = None

        if self.nu0 is None:
            if self.lastPoint == (k0, nu0, mu0, t0):
                lnDet = self.lnDet
            else:
                _sgn, lnDet = np.linalg.slogdet(T_dd)
            i = np.arange(1, d + 1)
            dnu0 = 0.5 * (d * np.log(t0) - lnDet + psi((1 - i + nu) / 2.).sum() - psi((1 - i + nu0) / 2.).sum())
        else:
            dnu0 = None

        if self.t0 is None:
            dt0 = 0.5 * (nu0 * d / t0 - nu * np.trace(Tinv_dd))
        else:
            dt0 = None

        if self.mu0 is None:
            D = dMu_d.reshape(-1, 1) + dMu_d.reshape(1, -1)
            dmu0 = -0.5 * nu * k0 * m / k * (Tinv_dd * D).sum()
        else:
            dmu0 = None

        return dk0, dnu0, dmu0, dt0

    def f(self, x):
        """
        perform the variable change
        """
        return -self.logMl(*self.convertVar(x))

    def df(self, x):
        """
        perform the variable change
        """
        k0, nu0, mu0, t0 = self.convertVar(x)
        dk0, dnu0, dmu0, dt0 = self.dLogMl(k0, nu0, mu0, t0)

        xL = []
        if dk0 is not None: xL.append(-dk0 * k0)
        if dnu0 is not None: xL.append(-dnu0 * (nu0 - self.d + 1))
        if dmu0 is not None: xL.append(-dmu0)
        if dt0 is not None: xL.append(-dt0 * t0)

        return np.array(xL)

    def opt(self):
        varL = [self.k0, self.nu0, self.mu0, self.t0]
        idxL = [i for i, var in enumerate(varL) if var is None]

        nameL = ['k0', 'nu0', 'mu0', 't0']

        if len(idxL) > 0:  # otherwise, there is no parameter to optimize

            print
            'mml for:', ', '.join([nameL[i] for i in idxL])

            x0 = np.zeros(len(idxL))

            allBounds = [
                (-10, 10),
                (-10, 10),
                (-1000, 1000),
                (-10, 10),
            ]

            boundL = [allBounds[i] for i in idxL]

            x, _f, d = fmin_l_bfgs_b(self.f, x0, self.df, bounds=boundL, approx_grad=False)

            #        ln_k0,ln_nu0,mu0,ln_t0 = x

            if self.verbose > 0:
                #                print 'Opt results : '
                #                print '-------------'
                #                for key, val in d.items():
                #                    print '%10s :'%key, val
                if d['warnflag'] != 0:
                    print
                    "WARNING : mml optimization of t warnflag = %d (%s)" % (d['warnflag'], d['task'])
                if d['funcalls'] > 100:
                    print
                    "WARNING : slow convergence for mml of t (%d func call)" % d['funcalls']


        else:
            if self.verbose > 0:
                print
                'all prior parameters are fixed ... no mml.'
            x = np.array([])
        return self.convertVar(x)

    def getPrior(self, avgLoss_d, stdLoss_dd, m):
        self.avgLoss_d = avgLoss_d
        self.stdLoss_dd = stdLoss_dd
        self.d = avgLoss_d.shape[0]
        self.m = m

        if self.nu0 is not None and self.nu0 < 0:
            self.nu0 = self.d - 1 - self.nu0

        k0, nu0, mu0, t0 = self.opt()

        if self.verbose > 0:
            for name, var in [('k0', k0), ('nu0', nu0), ('mu0', mu0), ('t0', t0)]:
                print
                '%10s : %.5g' % (name, var)

        mu0_d = np.ones(self.d) * mu0
        T0_dd = np.identity(self.d) * t0
        return k0, nu0, mu0_d, T0_dd


def randArgMin(val_n):
    val_n = np.asarray(val_n)
    minVal = np.min(val_n)
    idxL = np.where(minVal == val_n)[0]
    idx = random.choice(idxL)
    assert val_n[idx] == minVal
    return idx


#@Memoize_fs()
def getHpProb(avgLossSampler, nSample, miniBatch=1000):
    """
    converts the probability p(r|L) to p(h|S)
    :param avgLossSampler: an instance of HpProb
    :param nSample: number of samples from p(r|L)
    :param miniBatch: ?
    :return: p(h|S)
    """
    probBest = ProbBest()

    while nSample > 0:
        k = min(miniBatch, nSample)
        nSample -= k
        avgLoss_dk = avgLossSampler.sampleAvgLoss(k)
        probBest.updateFromLoss(avgLoss_dk)

    return probBest.getProb()


def getHpProb_oob(avgLossSampler, nSample, miniBatch=1000):
    probBest = ProbBest_oob()

    while nSample > 0:
        k = min(miniBatch, nSample)
        nSample -= k
        avgLoss_dk, free_samples_km = avgLossSampler.sampleAvgLoss_oob(k)
        probBest.updateFromLoss_oob(avgLoss_dk, free_samples_km)

    p_dm = probBest.getProb_oob()
    p_d = probBest.getProb()
    return p_d, p_dm


class HpProb:
    """
    a base class representing posterior probability
    """
    miniBatch = 1000
    nSample = 10000
    stack = True

    def getHpProb2L(self, loss_dnL):
        """
        just concatenate loss_dnL and treat it as a single layer.
        """
        #        if np.all(getattr(self, 'lastLoss', None) == loss_dnL):
        #            p_d = self.p_d
        #            print 'reusing last p_d'
        #        else:
        #

        if not hasattr(self, 'xv_idx'):
            loss_dm = np.hstack(tuple(loss_dnL))
        else:
            loss_dm = loss_dnL[self.xv_idx]

        p_d = self.getHpProb(loss_dm)
        #            self.p_d = p_d
        #            self.lastLoss = loss_dnL

        return p_d

    def getHpProb(self, loss_dm):
        """
        loss_dm is the loss matrix of size d*m, where d corresponds to the number of different hyper parameters 
        and m corresponds to the size of the dataset. More precisely, let h_i be the i^th predictor (out of d),
        (x_j, y_j), the j^th sample from the validation set (out of m) and let L be your favorite loss function, then
        looss_dm[i,j] = L( y_j, h_i(x_j) )  
        """
        self.setLoss(loss_dm)
        #        p_d = getHpProb(self, self.nSample, self.miniBatch)
        p_d = getHpProb(self, self.nSample, self.miniBatch)
        #        print unicodeHist( p_d-p_d_mp, 'p_d diff' )
        return p_d

    def getHpProb_oob(self, loss_dm):
        self.setLoss(loss_dm)
        return getHpProb_oob(self, self.nSample, self.miniBatch)


    def sampleAvgLoss(self, k):
        return self.sampleAvgLoss_oob(k)[0]


    def __call__(self, k):
        pb = ProbBest()
        risk_dk = self.sampleAvgLoss(k)
        pb.updateFromLoss(risk_dk)
        return pb.count_d

    def tweakParam(self, valInfoL, tstInfo=None):
        return self


def catLossInfo(lossInfoL):
    yTarget_mL = [lossInfo.yTarget_m for lossInfo in lossInfoL]
    yEstimate_dmL = [lossInfo.yEstimate_dm for lossInfo in lossInfoL]
    yTarget_m = np.hstack(tuple(yTarget_mL))
    yEstimate_dm = np.hstack(tuple(yEstimate_dmL))
    return LossInfo(yTarget_m, yEstimate_dm, lossInfoL[0].metric)


class LossInfo:
    def __init__(self, yTarget_m, yEstimate_dm, metric):
        assert yTarget_m.shape[0] == yEstimate_dm.shape[1]
        assert yTarget_m.ndim == 1
        assert yEstimate_dm.ndim == 2

        self.yTarget_m = yTarget_m
        self.yEstimate_dm = yEstimate_dm

        self.metric = metric

    def __str__(self):
        return 'LossInfo( shape = %s, metric=%s)' % (str(self.yEstimate_dm.shape), str(self.metric) )

    def getLoss(self):
        loss_dm = self.metric.loss(self.yTarget_m.reshape(1, -1), self.yEstimate_dm)
        assert loss_dm.shape == self.yEstimate_dm.shape
        return loss_dm

    def subset(self, idxL):
        self.yEstimate_dm = self.yEstimate_dm[idxL, :]

    def testDistr(self, p_d):

        if hasattr(p_d, 'predict'):
            yEstimate_m = p_d.predict(self.yEstimate_dm.T)
        else:

            assert p_d.shape[0] == self.yEstimate_dm.shape[0]
            if not (p_d >= 0).all():
                print
                'WARNING : negative probs.'
            #                print unicodeHist( p_d, 'p_d' )

            np.testing.assert_almost_equal(1., p_d.sum(0))

            yEstimate_m = self.metric.modelAveraging(self.yEstimate_dm, p_d)
        #        print unicodeHist( yEstimate_m, 'yEstimate_m' )
        return self.metric.loss(self.yTarget_m, yEstimate_m)


class HpProb2L:
    miniBatch = 1000
    nSample = 10000

    def getHpProb2L(self, loss_dnL):
        self.setLossL(loss_dnL)
        return getHpProb(self, self.nSample, self.miniBatch)


def getAvgLoss(loss_dnL):
    d = loss_dnL[0].shape[0]
    kFold = float(len(loss_dnL))

    avgLoss_d = np.zeros(d)
    for loss_dn in loss_dnL:
        assert loss_dn.shape[0] == d
        avgLoss_d += np.mean(loss_dn, 1) / kFold
    return avgLoss_d


class ArgMinXv:
    """
    the classical cross validation approach
    """

    def getHpProb2L(self, loss_dnL):
        if hasattr(self, 'xv_idx'):
            loss_dnL = [loss_dnL[self.xv_idx]]

        avgLoss_d = getAvgLoss(loss_dnL)
        idx = randArgMin(avgLoss_d)
        p_d = np.zeros(avgLoss_d.shape)
        p_d[idx] = 1.
        return p_d


    def __str__(self):
        return 'argMinXv'

    def short_name(self):
        return 'amin'


class SoftMax:
    def __init__(self, C, nTweak=20):
        self.C = C
        self.nTweak = nTweak

    def __str__(self):
        return 'Softmax(C=%.3g)' % (self.C)

    def short_name(self):
        return 'smin(twk=%d)' % self.nTweak

    def getHpProb(self, loss_dn):
        avgLoss_d = np.mean(loss_dn, 1)
        p_d = normalizeLogProb(-self.C * avgLoss_d)
        return p_d


    def getHpProb2L(self, loss_dnL):
        avgLoss_d = getAvgLoss(loss_dnL)
        p_d = normalizeLogProb(-self.C * avgLoss_d)
        return p_d

    def tweakParam(self, valInfoL, tstInfo=None):
        if self.nTweak <= 0:  return self
        samplerL = []
        for C in np.logspace(0, 5, self.nTweak):
            sampler = SoftMax(C)
            sampler.nSample = self.nSample
            samplerL.append(sampler)
        return paramTweaker(samplerL, valInfoL, tstInfo)


class BootstrapSubsampler(HpProb2L):
    def __init__(self, subSamplerTemplate, essr=1.):
        self.subSamplerTemplate = subSamplerTemplate
        self.essr = essr

    def __str__(self):
        return "Bootstrap (essr=%.3g) -> %s" % (self.essr, str(self.subSamplerTemplate))

    def setLossL(self, loss_dnL):
        self.subSamplerL = []
        for loss_dm in loss_dnL:
            subSampler = self.subSamplerTemplate.clone()
            subSampler.setLoss(loss_dm)
            self.subSamplerL.append(subSampler)

        self.n = len(self.subSamplerL)
        self.ess = int(round(self.essr * self.n))

    def getWeight(self, k):
        return bootStrapWeight(self.n, self.ess, k)

    def sampleAvgLoss(self, k=1000):
        w_nk = self.getWeight(k)

        avgLoss_dk = None
        for i, subSampler in enumerate(self.subSamplerL):
            w_1k = w_nk[i, :].reshape(1, -1)
            subAvgLoss_dk = subSampler.sampleAvgLoss(k)
            if avgLoss_dk is None:
                avgLoss_dk = subAvgLoss_dk * w_1k
            else:
                avgLoss_dk += subAvgLoss_dk * w_1k

        return avgLoss_dk


class EmpBayesPrior_T:
    def __init__(self, k0=1, nu0=-1, t_factor=1., verbose=0):
        self.k0 = k0
        self.nu0 = nu0
        self.verbose = verbose
        self.t_factor = t_factor

    def short_name(self):
        return 'emp'

    def emp_bayes_(self, loss_dm):

        batch = 10

        loss_n = loss_dm.flatten()  # returns a flattened copy
        np.random.shuffle(loss_n)
        k = np.floor(loss_n.size / batch)
        loss_kb = loss_n[:k * batch].reshape(k, batch)
        avgLoss_k = np.mean(loss_kb, 1)

        lower, median, upper = np.percentile(avgLoss_k, (15.9, 50, 84.1))

        self.mu0 = median
        self.t0 = ((upper - lower) / 2.) ** 2 * batch


        # keeps things sane
        if self.t0 <= 0.01:
            print
            '*** WARNING ***, t0 was %.3g, setting it to 0.01' % self.t0
            self.t0 = 0.01
        if self.t0 > 5:
            print
            '*** WARNING ***, t0 was %.3g, setting it to 5' % self.t0
            self.t0 = 5

    def emp_bayes(self, loss_dm):
        # keeps things sane

        self.mu0 = np.mean(loss_dm)
        self.t0 = np.var(loss_dm)

        if self.t0 <= 0.01:
            print
            '*** WARNING ***, t0 was %.3g, setting it to 0.01' % self.t0
            self.t0 = 0.01
        if self.t0 > 5:
            print
            '*** WARNING ***, t0 was %.3g, setting it to 5' % self.t0
            self.t0 = 5


    def getPrior(self, avgLoss_d, stdLoss_dd, m):

        d = len(avgLoss_d)

        nu0 = self.nu0
        if self.nu0 < 0:
            nu0 = d - 1 - nu0
        else:
            assert nu0 >= d - 1

        if self.verbose > 0:
            for name, var in [('k0', self.k0), ('nu0', nu0), ('mu0', self.mu0), ('t0', self.t0)]:
                print
                '%10s : %.5g' % (name, var)

        mu0_d = np.ones(d) * self.mu0
        T0_dd = np.identity(d) * self.t0 * self.t_factor
        return self.k0, nu0, mu0_d, T0_dd


class LossSampler_t(HpProb):
    def __init__(self, prior, essr=1., nTweak=0, keep_ratio=1.):
        """
        prior: either MML_tFull or EmpBayesPrior_T
        """
        self.prior = prior
        self.essr = essr
        self.nTweak = nTweak
        self.keep_ratio = keep_ratio

    def __str__(self):
        if not hasattr(self, 'keep_ratio'):
            self.keep_ratio = 1.
        return "tSampler (essr= %.2f, prior= %s, keep=%.1f )" % (self.essr, str(self.prior), self.keep_ratio)

    def short_name(self):
        return "t(r=%.1f,twk=%d,p=%s)" % (self.essr, self.nTweak, self.prior.short_name() )

    def clone(self):
        sampler = LossSampler_t(self.prior, self.essr, keep_ratio=self.keep_ratio)
        if hasattr(self, "nSample"):
            sampler.nSample = self.nSample
        return sampler

    def getPosterior(self, loss_dm):

        d, m = loss_dm.shape
        ess = self.essr * m  # effective sample size
        avgLoss_d, stdLoss_dd, m = tSufficientStats(loss_dm)

        k0, nu0, mu0_d, T0_dd = self.prior.getPrior(avgLoss_d, stdLoss_dd, m)
        print('k0=%.1f, nu0=%.1f, mu0=%.2f, t0=%.2f' % (k0, nu0, mu0_d[0], T0_dd[0, 0]))

        k, nu, mu_d, T_dd = tPosterior(k0, nu0, mu0_d, T0_dd, avgLoss_d, stdLoss_dd, ess, m)

        assert mu_d.shape == (d,)
        assert T_dd.shape == (d, d)

        return mu_d, T_dd, k, nu


    def setLossInfo(self, lossInfo):
        pass


    def set_mask(self, loss_dm):
        self.avg_loss_d = np.mean(loss_dm, 1)
        threshold = np.percentile(self.avg_loss_d, self.keep_ratio * 100)
        self.mask_d = self.avg_loss_d > threshold
        print
        'keep ratio : %.3g, mean mask : %.3g' % (self.keep_ratio, np.mean(self.mask_d))


    def setLoss(self, loss_dm):
        d, _m = loss_dm.shape

        self.set_mask(loss_dm)

        if hasattr(self.prior, 'emp_bayes'):
            self.prior.emp_bayes(loss_dm[~self.mask_d, :], )

        self.mu_d, T_dd, kappa, nu = self.getPosterior(loss_dm)
        self.dFree = nu - d + 1
        self.S_dd = T_dd / kappa / self.dFree

    def sampleAvgLoss(self, k):
        avgLoss_dk = multivariate_t_rvs(self.mu_d, self.S_dd, self.dFree, k).T
        avgLoss_dk[self.mask_d, :] = self.avg_loss_d[self.mask_d].reshape(-1,
                                                                          1)  # use a variance of zero with bad predictors, to avoid the misspecification to screw everything

        if np.any(np.isnan(avgLoss_dk)):
            print
            'nan occured with %s' % (str(self))
            isnan_k = np.any(np.isnan(avgLoss_dk), 0)
            print
            '%d/%d samples with nan' % (np.sum(isnan_k), k)
            print
            'dFree : %.3g' % self.dFree

            eS_dd = np.linalg.eigvalsh(self.S_dd)
            #             print unicodeHist(eS_dd, 'eS_dd')
            import sys
            import time

            sys.stdout.flush()
            time.sleep(10)

        return avgLoss_dk


    def tweakParam(self, valInfoL, tstInfo=None):
        if self.nTweak <= 0: return self
        samplerL = []

        essrL = np.logspace(-2, 0, self.nTweak)
        for essr in essrL:
            sampler = self.clone()
            sampler.essr = essr
            samplerL.append(sampler)

        #        prior = (essrL - self.essr) ** 2 * 1e-6
        return paramTweaker(samplerL, valInfoL, tstInfo)


class PriorMeasure:
    def stick_break_avg(self, alpha, k):


        if alpha > 100:  # randn is a really good approximation for high values of alpha
            mean, std = self.mean_std()
            std /= np.sqrt(alpha)
            return np.random.randn(self.d, k) * std + mean

        n = max(20, int(alpha * 10))

        prior_sample_dk = np.empty((self.d, k))

        for i in range(k):
            w_n = stick_breaking(alpha, n, 1).flatten()
            measure_sample_nd = self.sample(n)
            prior_sample_dk[:, i] = np.dot(w_n, measure_sample_nd)

        return prior_sample_dk

    def set_shape(self, d):
        self.d = d

    def empirical_bayes(self, loss_dm):
        pass


class NormalMeasure(PriorMeasure):
    def __init__(self, mu=0, var=1.):
        self.mu = mu
        self.s = np.sqrt(var)

    def sample(self, n):
        return np.random.randn(n, self.d) * self.s + self.mu


class SymmetricBernoulli(PriorMeasure):
    def __init__(self, q=0.5):
        self.q = q  # probability of error

    def empirical_bayes(self, loss_dm):
        """Estimate the prior parameters from the observations"""
        self.q = np.mean(loss_dm)

    def mean_std(self):
        return self.q, np.sqrt(self.q * (1 - self.q))

    def sample(self, n):
        return (np.random.rand(n, self.d) < self.q).astype(np.int)


class GammaMeasure(PriorMeasure):
    def __init__(self, k=None, theta=None):
        self.k = k
        self.theta = theta

    def empirical_bayes_(self, loss_dm):
        """Estimate the prior parameters from the observations"""
        mu = np.mean(loss_dm)
        std = np.sqrt(np.mean(np.var(loss_dm, 1)))

        self.k = (mu / std) ** 2
        self.theta = std ** 2 / mu

    #         print 'empirical Bayes : k = %.3g, theta=%.3g'%(self.k, self.theta)

    def empirical_bayes(self, loss_dm):
        x = loss_dm.flatten()

        epsilon = 1e-8
        assert np.all(x > -epsilon)  # makes sure there is no negative number
        x[x < epsilon] = epsilon  # zeros are ok but are removed to avoid log(0)

        # we use the formula found on wikipedia to estimate k and theta
        s = np.log(np.mean(x)) - np.mean(np.log(x))
        if self.k is None:
            self.k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12. * s)
        if self.theta is None:
            self.theta = np.mean(x) / self.k

        print
        'Gamma Measure : k = %.3g, theta=%.3g' % (self.k, self.theta)


    def mean_std(self):
        mean = self.k * self.theta
        std = np.sqrt(self.k) * self.theta
        return mean, std

    def sample(self, n):
        return np.random.gamma(self.k, self.theta, (n, self.d))


def _stick_break(v_nk):
    pi_nk = v_nk
    pi_nk[1:, :] *= np.cumprod(1 - v_nk[:-1, :], axis=0)
    return pi_nk


def stick_breaking(alpha, n, k):
    v_nk = np.random.beta(1, alpha, (n, k))
    return _stick_break(v_nk)


class LossSampler_Dirichlet(HpProb):
    def __init__(self, essr=1., alpha=10, mRatio=0, prior_measure=None, nTweak=0, nTweakAlpha=0):
        """
        Compute the hyperparameter posterior using the Dirichlet distribution
        
        if alpha < 1. : max( 1,  alpha = m * self.alpha )
                    
        """

        self.essr = essr
        self.alpha = alpha
        self.mRatio = mRatio

        self.prior_measure = prior_measure
        self.nTweak = nTweak
        self.nTweakAlpha = nTweakAlpha


    def short_name(self):
        return "D(r=%.1f,twk=%d,a=%.1f)" % (self.essr, self.nTweak, self.alpha )

    def clone(self):
        return LossSampler_Dirichlet(self.essr, self.alpha, self.mRatio, self.prior_measure)


    def setLoss(self, loss_dm):

        self.loss_dm = loss_dm

        if self.prior_measure is None:
            if set(np.unique(loss_dm)) == set((0, 1)):
                print
                'Selecting SymmetricBernoulli'
                self.prior_measure = SymmetricBernoulli()
            else:
                print
                'Selecting GammaMeasure'
                self.prior_measure = GammaMeasure()

        self.prior_measure.set_shape(loss_dm.shape[0])
        self.prior_measure.empirical_bayes(loss_dm)

    def sampleAvgLoss_oob(self, k):


        d, m = self.loss_dm.shape
        ess = self.essr * m
        alpha = self.alpha + ess * self.mRatio
        free_samples_km = np.ones((k, m))


        # the prior
        avgLoss_prior_dk = self.prior_measure.stick_break_avg(alpha, k)

        if m == 0:
            return avgLoss_prior_dk, free_samples_km


        # the mixture between the prior and the posterior
        beta_1k = np.random.beta(alpha, ess, k).reshape(1, -1)

        #        beta_1k = np.zeros((1,k))+alpha/ess



        # the posterior
        weight_mk = np.random.dirichlet(np.ones(m) * self.essr, k).T
        assert weight_mk.shape == (m, k)
        np.testing.assert_almost_equal(np.sum(weight_mk, 0), 1)

        avgLoss_posterior_dk = np.dot(self.loss_dm, weight_mk)
        assert avgLoss_posterior_dk.shape == (d, k)

        avgLoss_dk = beta_1k * avgLoss_prior_dk + (1 - beta_1k) * avgLoss_posterior_dk

        if verbosity >= 3:
            from graalUtil.num import uHist

            data = {
                'k': k,
                'alpha': alpha,
                #                "beta_1k":uHist(beta_1k,'beta'),
                #                "free_samples_km":uHist(free_samples_km, 'free_samples'),
                'avgLoss_prior_dk': uHist(avgLoss_prior_dk, 'avgLoss_prior_dk'),
                'avgLoss_posterior_dk': uHist(avgLoss_posterior_dk, 'avgLoss_posterior_dk'),
            }
            append_to_dump('sampleAvgLoss_oob', data)

        return avgLoss_dk, free_samples_km


    def __str__(self):
        return 'Dirichlet (alpha=%.3g, essr=%.3f, nTweak=%d)' % (self.alpha, self.essr, self.nTweak)


    def _get_param_list(self):
        import itertools

        if self.nTweakAlpha > 0:
            alphaL = np.logspace(0, 2, self.nTweakAlpha)
        else:
            alphaL = np.array([self.alpha])

        if self.nTweak > 0:
            essrL = np.logspace(-2, 0, self.nTweak)
        else:
            essrL = np.array([self.essr])

        return list(itertools.product(essrL, alphaL))


    #
    def _double_grid(self):

        samplerL = []

        paramL = self._get_param_list()
        for essr, alpha in paramL:
            sampler = self.clone()
            sampler.essr = essr
            sampler.alpha = alpha
            sampler.nSample = self.nSample
            samplerL.append(sampler)

        return samplerL, paramL


    #
    def tweakParam(self, valInfoL, tstInfo=None):
        print
        'tweakParam'
        if self.nTweak == 0 and self.nTweakAlpha == 0: return self

        samplerL, paramL = self._double_grid()

        #        prior = (paramL - getattr(self, paramName) )** 2 * 1e-6
        #        prior = np.zeros(len(paramL))
        #        prior =_centered_prior(len(paramL))

        return paramTweaker_oob(samplerL, valInfoL, tstInfo, paramL=paramL)


class LossSampler_Dirichlet_old(HpProb):
    def __init__(self, essr=1., Nalpha=None, alphaRatio=1, valueL=None, nTweak=0):
        """
        Compute the hyperparameter posterior using the Dirichlet distribution
        
        Parameters
        ----------        
        
        essr : effective sample size ratio. Put a value smaller than 1 when you think  that the 
            counts are correlated
            
        alpha : the prior distribution is a Dirichlet of parameter (alpha, alpha, ..., alpha ).
            If the value is left to None, alpha is evaluated from empirical Bayes. 
        
        alphaRatio : when alpha is obtained from empirical Bayes, this ratio multiplies the obtained alpha*.

        valueL : the list of possible loss values. If None, it is inferred from G1. In the case of
            the zero one loss metric, the possible values are : [0,1]. 
        """

        self.essr = essr
        self.Nalpha = Nalpha
        self.alphaRatio = alphaRatio
        self.valueL = valueL
        self.nTweak = nTweak


    def clone(self):
        return LossSampler_Dirichlet(self.essr, self.Nalpha, self.alphaRatio, self.valueL)

    def __str__(self):
        return 'Dirichlet(essr=%.2f, Nalpha=%.3g, %d sample)' % (self.essr, self.Nalpha, self.nSample)

    def getCount(self, loss_dm):
        d = loss_dm.shape[0]
        countD = {}
        for loss_d in loss_dm.T:
            assert loss_d.shape[0] == d
            key = tuple(loss_d)
            if key in countD:
                countD[key] += 1
            else:
                countD[key] = 1

        G1_dn, k_n = zip(*countD.items())
        #        print unicodeHist(k_n, 'count')

        return np.array(G1_dn).T, np.array(k_n)

    def setLoss(self, loss_dm):

        self.G1_dn, self.k_n = self.getCount(loss_dm)

        assert self.G1_dn.shape[1] == self.k_n.shape[0]

        d, n = self.G1_dn.shape

        if self.valueL is None:
            self.valueL = np.unique(self.G1_dn)

        self.N = float(len(self.valueL)) ** d  # total number of states of the dirichlet space

        if self.Nalpha is None:  # find this parameter through empirical bayes
            print
            'searching for alpha'
            Nalpha = BetaLogLike(self.N, self.k_n).getAlphaStar() * self.alphaRatio
        else:
            Nalpha = self.Nalpha


        #        print 'Nalpha : %.3g' % (Nalpha)

        alpha = Nalpha / self.N
        #        alphaSum = (self.N - n) * alpha
        alphaSum = Nalpha - n * alpha

        self.qParam = np.array(list(self.k_n * self.essr + alpha) + [alphaSum])
        self.rParam = np.array([alpha] * n + [alphaSum])

    #        print unicodeHist(self.qParam, 'qParam')
    #        print unicodeHist(self.rParam, 'rParam')

    #        print 'qParam : ', self.qParam
    #        print 'rParam : ', self.rParam

    def sampleGr(self, k):
        d = self.G1_dn.shape[0]
        l = len(self.valueL)
        N_ = self.N / l
        dir_dkl = np.random.dirichlet([N_] * l, (d, k))
        Gr_dk = np.dot(dir_dkl, self.valueL)
        #        print unicodeHist(Gr_dk, 'Gr_dk')
        return Gr_dk

    def sampleAvgLoss(self, k):

        q_kn = np.random.dirichlet(self.qParam, k)
        r_kn = np.random.dirichlet(self.rParam, k)
        Gr_dk = self.sampleGr(k)

        qBar_k1 = q_kn[:, -1].reshape(-1, 1)
        q_kn = q_kn[:, :-1]
        rBar_k1 = r_kn[:, -1].reshape(-1, 1)
        r_kn = r_kn[:, :-1]

        diff_kn = q_kn / qBar_k1 - r_kn / rBar_k1

        Gq_dk = (np.dot(self.G1_dn, diff_kn.T) + Gr_dk / rBar_k1.T) * qBar_k1.T

        return Gq_dk


    def tweakParam_essr(self, valInfoL, tstInfo=None):
        if self.nTweak <= 0: return self
        samplerL = []

        essrL = np.linspace(0.01, 0.5, self.nTweak)
        for essr in essrL:
            sampler = LossSampler_Dirichlet(essr, self.Nalpha)
            sampler.nSample = self.nSample
            samplerL.append(sampler)

        prior = (essrL - self.essr) ** 2 * 1e-6
        return paramTweaker(samplerL, valInfoL, tstInfo, prior)

    def tweakParam(self, valInfoL, tstInfo=None):
        if self.nTweak <= 0: return self
        samplerL = []

        NalphaL = np.logspace(10, 20, self.nTweak)
        for Nalpha in NalphaL:
            sampler = LossSampler_Dirichlet(self.essr, Nalpha)
            sampler.nSample = self.nSample
            samplerL.append(sampler)

        #        prior = (NalphaL - self.Nalpha) ** 2 * 1e-6

        prior = np.ones(self.nTweak)
        return paramTweaker(samplerL, valInfoL, tstInfo, prior)


def bootStrapWeight(n, mSample, batchSize=50):
    """
    n : number of states
    
    mSample : number of samples to draw
    
    batchSize : number of repeat
    
    returns : w_nb, a matrix of weight of shape n,b where b is the batchSize
    """

    w_nb = np.zeros((n, batchSize))

    for i in range(batchSize):
        count = np.bincount(np.random.randint(0, n, mSample))
        #        w_nb[:, i] = np.bincount(np.random.randint(0, n, mSample), minlength=n) # only available on most recent version of numpy
        w_nb[:len(count), i] = count
    #    print unicodeHist( w_nb )
    return w_nb / float(mSample)


def bootStrapWeight_(n, mSample, batchSize=50):
    """
    n : number of states
    
    mSample : number of samples to draw
    
    batchSize : number of repeat
    
    returns : w_nb, a matrix of weight of shape n,b where b is the batchSize
    """

    w_nL = [np.bincount(np.random.randint(0, n, mSample), minlength=n) for _i in range(batchSize)]
    w_nb = np.array(w_nL).T / float(mSample)
    assert w_nb.shape == (n, batchSize)
    return w_nb


def bootStrapWeight_multinomial(n, mSample, batchSize=50):
    "slower"
    p = 1. / n
    w_nb = np.random.multinomial(mSample, (p,) * n, batchSize).T
    assert w_nb.shape == (n, batchSize)
    return w_nb / float(mSample)


def paramTweaker(lossSamplerL, valInfoL, tstInfo=None, prior=None):
    valInfo = catLossInfo(valInfoL)
    loss_dm = valInfo.getLoss()

    valRiskL = []
    tstRiskL = []
    essrL = []
    for lossSampler in lossSamplerL:
        essrL.append(getattr(lossSampler, 'essr', None))
        p_d = lossSampler.getHpProb(loss_dm)
        valRisk = np.mean(valInfo.testDistr(p_d))
        valRiskL.append(valRisk)

        if tstInfo is not None:
            tstRisk = np.mean(tstInfo.testDistr(p_d))
            tstRiskL.append(tstRisk)
        else:
            tstRisk = np.NAN

        if verbosity > 1:
            print
            '%-7.5f vs %-7.5f (%s)' % (valRisk * 100, tstRisk * 100, str(lossSampler))

    valRiskL = np.asarray(valRiskL)
    if prior is not None:
        valRiskL += prior
    #    valRiskL += np.random.rand(valRiskL.shape[0]) * 1e-6

    idx = randArgMin(valRiskL)
    lossSampler = lossSamplerL[idx]
    print
    'chose %s' % str(lossSampler)

    if verbosity > 2:
        from matplotlib import pyplot as pp

        essr = essrL[idx]
        mn = np.min((valRiskL, tstRiskL))
        mx = np.max((valRiskL, tstRiskL))
        pp.plot(essrL, valRiskL, '.-', label='val')
        pp.plot(essrL, tstRiskL, '.-', label='tst')
        #        pp.plot( essrL, prior, '.-', label='prior' )
        pp.plot([essr, essr], [mn, mx], '-')
        pp.legend(loc='best')
        pp.show()

    return lossSampler


def append_to_dump(tag, data=None):
    import time as t

    timestamp = t.time()

    from cPickle import dump

    try:
        import mpi4py.MPI as MPI  #@UnresolvedImport

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0

    fh = open('obj_dump_%d.pklL' % rank, 'a')
    dump((rank, timestamp, tag, data), fh, 2)
    fh.close()


def _centered_prior(n):
    x = np.linspace(-1, 1, n)
    return 1e-8 * x ** 2


def paramTweaker_oob(lossSamplerL, valInfoL, tstInfo=None, prior=None, paramL=None):
    print
    'starting paramTweaker'

    valInfo = catLossInfo(valInfoL)
    loss_dm = valInfo.getLoss()

    valRiskL = []
    tstRiskL = []
    for lossSampler in lossSamplerL:
        print
        'Testing %s' % (str(lossSampler))
        p_d, p_dm = lossSampler.getHpProb_oob(loss_dm)
        valRisk = np.mean(valInfo.testDistr(p_dm))
        valRiskL.append(valRisk)

        if tstInfo is not None:
            tstRisk = np.mean(tstInfo.testDistr(p_d))
            tstRiskL.append(tstRisk)

    if verbosity >= 2:
        data = {
            'valRiskL': valRiskL,
            'tstRiskL': tstRiskL,
            'lossSampler': lossSamplerL[0],
            'prior': prior,
            'paramL': paramL}
        append_to_dump('paramTweaker_oob', data)

    valRiskL = np.asarray(valRiskL)
    if prior is not None:
        valRiskL += prior
    #    valRiskL += np.random.rand(valRiskL.shape[0]) * 1e-6

    idx = randArgMin(valRiskL)
    lossSampler = lossSamplerL[idx]
    print
    'chose %s' % str(lossSampler)

    return lossSampler


class LossSampler_bootstrap(HpProb):
    def __init__(self, essr=1., nTweak=0):
        self.essr = essr
        self.nTweak = nTweak

    def clone(self):
        return LossSampler_bootstrap(self.essr)

    def short_name(self):
        return 'b(r=%.1f,twk=%d)' % (self.essr, self.nTweak)

    def __str__(self):
        return 'bootstrapSampler (essr=%.2f, nTweak=%d)' % (self.essr, self.nTweak)

    def setLoss(self, loss_dm):
        self.loss_dm = loss_dm
        self.m = loss_dm.shape[1]
        self.ess = int(np.ceil(self.m * self.essr))


    def sampleAvgLoss_oob(self, k):
        #        print self.m, self.ess, k
        w_mk = bootStrapWeight(self.m, self.ess, k)
        free_samples_km = (w_mk.T == 0) * 1
        return np.dot(self.loss_dm, w_mk), free_samples_km


    def tweakParam(self, valInfoL, tstInfo=None):
        '''
        seems to be a grid search ability to set the parameters empirically
        :param valInfoL:
        :param tstInfo:
        :return:
        '''
        if self.nTweak <= 0:
            return self
        samplerL = []

        essrL = np.logspace(-2, 0, self.nTweak)
        for essr in essrL:
            sampler = LossSampler_bootstrap(essr)
            sampler.nSample = self.nSample
            samplerL.append(sampler)

        #        prior = (essrL - self.essr) ** 2 * 1e-10
        prior = _centered_prior(len(essrL))
        return paramTweaker_oob(samplerL, valInfoL, tstInfo, prior, essrL)


class LossSamplerBB(HpProb):
    def __init__(self, essr=1., noise=0., nTweak=0):
        self.essr = essr
        self.noise = noise
        self.nTweak = nTweak

    def __str__(self):
        return 'bbSampler (essr=%.2f, noise=%.3g, nSample=%d)' % (self.essr, self.noise, self.nSample)


    def clone(self):
        return LossSamplerBB(self.essr, self.noise, self.nTweak)

    def setLoss(self, loss_dm):
        self.loss_dm = loss_dm
        m = loss_dm.shape[1]
        self.alpha_m = np.ones(m) * self.essr

    def sampleAvgLoss(self, k):
        w_mk = np.random.dirichlet(self.alpha_m, k).T
        avgLoss_dk = np.dot(self.loss_dm, w_mk)
        if self.noise > 0:
            avgLoss_dk += np.random.randn(*avgLoss_dk.shape) * self.noise
        return avgLoss_dk

    def tweakParam(self, valInfoL, tstInfo=None):
        if self.nTweak <= 0: return self
        samplerL = []

        essrL = np.linspace(0.01, 0.5, self.nTweak)
        for essr in essrL:
            sampler = LossSamplerBB(essr)
            sampler.nSample = self.nSample
            samplerL.append(sampler)

        prior = (essrL - self.essr) ** 2 * 1e-6
        return paramTweaker(samplerL, valInfoL, tstInfo, prior)


def resample(loss_dm, m_=100):
    idx = np.random.randint(0, loss_dm.shape[1], m_)
    print(np.bincount(idx))
    return loss_dm[:, idx]


def testDerivative(loss_dm, x=None):
    from matplotlib import pyplot as pp

    mml = MML_tFull(*tSufficientStats(loss_dm))

    if x is None:
        x = np.array([0., 2., 0., 0.])

    xiL = np.linspace(-10, 20, 1000)
    idx = 0
    yL = []
    dyL = []
    for xi in xiL:
        x[idx] = xi
        y = mml.f(x)
        #        print y
        yL.append(y)
        dyL.append(mml.df(x)[idx])

    pp.plot(xiL, yL)

    y_L = np.cumsum(dyL) * np.mean(np.diff(xiL))
    pp.plot(xiL, y_L + yL[0])

    pp.show()


def plotAroundPoint(mml, x=None):
    from matplotlib import pyplot as pp

    if x is None:
        x = np.array([0, 0, 0, 0])

    dL = np.linspace(-15, 15, 1000)

    nameL = ['ln_k0', 'ln_nu0', 'mu0', 'ln_t0']

    for i in range(4):
        yL = []
        xL = []
        for d in dL:
            x_ = np.copy(x)
            x_[i] += d
            yL.append(mml.f(x_))
            xL.append(x_[i])
        pp.figure()
        pp.plot(xL, yL, label=nameL[i])
        pp.legend(loc='best')

    pp.show()


def checkGrad(loss_dm):
    from scipy.optimize import check_grad
    from numUtil import unicodeHist
    from matplotlib import pyplot as pp

    loss_dm = np.vstack((loss_dm,) * 2)
    loss_dm = np.hstack((loss_dm,) * 2)
    n = 1000
    mml = MML_tFull(*tSufficientStats(loss_dm))

    pt_n4 = np.random.randn(n, 4) * 4

    errL = []
    bigErrL = []
    for pt in pt_n4:
        e = check_grad(mml.f, mml.df, pt)
        if e > 1e-3:
            print(e, pt)
            bigErrL.append(pt)
        errL.append(e)

    print()
    unicodeHist(errL, 'grad err distr')
    bigErrL = np.array(bigErrL)
    #    errL = np.array(errL)
    idx1 = 2
    idx2 = 0
    pp.plot(pt_n4[:, idx1], pt_n4[:, idx2], '.b', markersize=1.)
    pp.plot(bigErrL[:, idx1], bigErrL[:, idx2], '.r')
    pp.show()


def minimizeMml(loss_dm):
    d = loss_dm.shape[0]
    nu0 = d - 1 + 1e-0
    #    nu0= None
    mml = MML_tFull(k0=1e-5, nu0=nu0)
    k0, nu0, mu0_d, T0_dd = mml.getPrior(*tSufficientStats(loss_dm))
    mu0 = mu0_d[0]
    t0 = T0_dd[0, 0]

    print()
    print('k0 = %.3g' % k0)
    print('nu0 = %.3g' % nu0)
    print('mu0 = %.3g' % mu0)
    print('t0 = %.3g' % t0)

    x = mml.invConvertVar(k0, nu0, mu0, t0)

    print('x : ', x)

    mml.k0 = None
    mml.mu0 = None
    mml.nu0 = None
    plotAroundPoint(mml, x)

    return x


def multivariateNormalLoss(d=10, m=1000):
    A = np.random.rand(d, d)
    S = np.dot(A, A.T)
    mu = np.random.randn(d) * 0.001 + 10
    print
    'meam mu : %.3g' % (np.mean(mu))
    #     print unicodeHist(np.linalg.eigvalsh(S), 'eig val of S')
    loss_dm = np.random.multivariate_normal(mu, S, m).T

    return loss_dm


#pool = Pool(4)


def test_softmin():
    val = np.array([0.1, 0.1, 0.10000001, 0.11])
    for p in softmin(val, alpha=1e8):
        print
        '%.5f' % p


if __name__ == "__main__":
    #  x = minimizeMml(loss_dm)
    #  testMethods()
    test_softmin()
