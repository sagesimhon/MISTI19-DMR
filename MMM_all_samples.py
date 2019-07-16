import numpy as np
from scipy.special import logsumexp
import json

class MMM:
    def __init__(self, k, e=None, pi=None, n=None, m=None, epsilon=1e-4, max_iter=10000): # if e is not specified, then n,m must be
        self.k = k
        if e is None:
            self.n = n
            self.m = m
            self.e = np.random.rand(self.n, self.m)
        else:
            self.n = e.shape[0]
            self.m = e.shape[1]
            self.e = e
        if pi is None:
            self.pi = np.random.rand(self.k, self.n)
        else:
            self.pi = pi
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.__normalize()
        self.pi_log = np.log(self.pi)
        self.e_log = np.log(self.e)


    def __normalize(self):
        self.pi /= self.pi.sum()

        for row in range(len(self.e)):
            self.e[row] /= self.e[row].sum()


    def __update_model(self, E_log, A_log):
        self.pi_log = A_log
        self.e_log = E_log

    def expectation(self, B_log):
        # B, A and E are in log domain
        E_log = np.log(np.zeros((self.n, self.m)))  # initialize k by n by m array
        A_log = np.zeros((self.k, self.n))  # initialize length-m vector.
        for id in range(self.k):
            tmp = (self.pi_log[id] + self.e_log.T).T # log prob of topic-word (nxm)
            normalizer_log = logsumexp(tmp, 0) # log prob of specific word (sum over all topics) (m)

            sample_E_log = B_log[id] + tmp - normalizer_log # expectation of each topic-word in 1 sample (nxm)
            A_log[id] = logsumexp(sample_E_log, 1) # expectation of each topic in 1 sample (n)
            E_log = np.logaddexp(sample_E_log, E_log) # expectation of each topic-word in all samples (nxm)

        return E_log, A_log

    # M step
    def maximize(self, E_log, A_log):
        #model = Model(len(A_log), len(E_log[0]))
        A_log -= logsumexp(A_log, 1, keepdims=True)
        E_log -= logsumexp(E_log, 1, keepdims=True)
        # E_log = self.e_log

        #for i in range(model.n):
        #    model.pi_log[i] = A[i] - logsumexp(A)
        #normalizer = logsumexp(E, 1) #sum(E[i][k] for k in range(model.m))
        #for j in range(model.m):
        #    model.e_log[i][j] = E[i][j]/normalizer
        return E_log, A_log


    def refit(self, X):
        return


    def fit(self, X):
        categories = np.zeros((self.k, self.m))
        B = np.zeros((self.k, self.m))
        _, m = self.e.shape

        for id, key in enumerate(X.keys()):
            categories, counts = np.unique(X[key]['sequence'], return_counts=True)
            tmp = np.zeros(m)
            tmp[categories] = counts
            B[id] = tmp
        error = 10000
        estimate = self.log_likelihood(B)
        B_log = np.log(B)
        iters = 0
        while error > self.epsilon and iters < self.max_iter:
            E_log, A_log = self.expectation(B_log)
            E_log, A_log = self.maximize(E_log, A_log)
            self.__update_model(E_log, A_log)
            new_estimate = self.log_likelihood(B)
            error = new_estimate - estimate
            estimate = new_estimate
            iters += 1
            print(new_estimate)
        return estimate


    def log_likelihood(self, B):
        res = 0
        for i in range(self.k):
            res += np.sum(B[i] * logsumexp(self.pi_log[i] + self.e_log.T, 1))

        return res



def load_test_files(sequence_data_filename='simple_data/data_for_michael.json', e_filename='simple_data/signatures_for_michael.npy'):
    """ Load and parse DNA database """
    with open(sequence_data_filename, 'r') as f:
        data = json.load(f)
    e = np.load(e_filename)
    return e, data




def run_MMM(test_sample_keys=None):
    e, data = load_test_files()
    total = 0
    total_keys = str(len(data.keys()))
    print('total keys: ' + total_keys)
    i = 0
    for k in test_sample_keys or data.keys():
        model = MMM(560, e)
        total += model.fit(data)
        print('processed ' + str(i) + ' / ' + total_keys)
        i += 1

    print('----total---')
    print(total)


def main():
    run_MMM()



if __name__ == "__main__" or __name__ == 'basic_model_MMM':
    main()
