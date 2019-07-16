import numpy as np
from scipy.special import logsumexp
import json
# E step
def expectation(B_log, model):
    # B, A and E are in log domain
    E = np.zeros((model.n, model.m)) #initialize n by m array
    A = np.empty(model.n) #initialize length-n vector. topic freqs
    normalizer_log = np.zeros(model.m) #initialize length-m vector.
    for j in range(model.m):
        normalizer_log[j] = logsumexp([model.pi_log[k] + model.e_log[k][j] for k in range(model.n)])

    for i in range(model.n): #for topic i
        for j in range(model.m): #for word j
            E[i][j] = (B_log[j] + (model.pi_log[i]+model.e_log[i][j])) - normalizer_log[j]

    A = logsumexp(E, 1)

    return E, A

# M step
def maximize(E, A):
    # model = Model(len(A), len(E[0]))
    A -= logsumexp(A)
    # for i in range(model.n):
    #     model.pi_log[i] = A[i] - logsumexp(A)
        # normalizer = logsumexp(E, 1) #sum(E[i][k] for k in range(model.m))
        # for j in range(model.m):
        #     model.e_log[i][j] = E[i][j]/normalizer
    return A


def EM(epsilon=1e-4, n=10000):
    X, e, ids = util_function()
    categories, B = np.unique(X, return_counts=True)
    _, m = e.shape
    tmp = np.zeros(m)
    tmp[categories] = B
    B = tmp
    error = 10000
    model = Model(e)
    estimate = get_estimate(model, B)
    B_log = np.log(B)
    while error > epsilon:
        E, A = expectation(B_log, model)
        A = maximize(E, A)
        model.pi_log = A
        new_estimate = get_estimate(model, B)
        error = new_estimate - estimate
        estimate = new_estimate
        print(new_estimate)

    return model


def get_estimate(model, B): # ???
    res = np.sum(B * logsumexp(model.pi_log + model.e_log.T, 1))

    return res

def util_function():
    """ Load and parse DNA database """
    filename = 'simple_data/data_for_michael.json'
    with open(filename, 'r') as f:
        data = json.load(f)

    e = np.load('simple_data/signatures_for_michael.npy')
    X = np.array(data['PD10010']['sequence'])
    return X, e, data.keys # todo: e

class Model:
    def __init__(self, e):
        n, m = e.shape
        self.pi = np.random.rand(n)
        self.e = e
        self.normalize()
        self.n = n #number of topics
        self.m = m #number of words

        self.pi_log = np.log(self.pi)
        self.e_log = np.log(self.e)

    def normalize(self):
        self.pi /= self.pi.sum()

        for row in range(len(self.e)):
            self.e[row] /= self.e[row].sum()


if __name__ == "__main__":
    EM()
