import numpy as np
from scipy.special import logsumexp
from scipy.special import gamma
from scipy import optimize as optimize
from scipy.special import digamma as digamma
import json
import math

class DMR:
    """Topic Model with DMR"""
    def __init__(self, phi, x, mu=0, sigma=1, epsilon=1e-4, max_iter=1e4):
        # usually T=12, F=0, D=560, m=96
        self.T, self.m = phi.shape # # topics and # mutation categories
        self.D, self.F = x.shape # # samples/documents and # features in metadata

        ###### Given #######
        self.mu = mu  # mean of lambda's dist
        self.sigma = sigma  # variance of lambda's dist

        self.phi = phi # (e from MMM) - signatures - T by m
        temp = np.ones((self.D, self.F + 1))
        temp[:,1:] = x
        self.x = temp # metadata - D by F with column of all 1s
        ####################

        self.alpha = None # weights for Dir dist - D by T
        self.lam = np.empty((self.T, self.F + 1))
        self.z = None # np.empty((self.D, self.n_d))?

        if self.lam == np.empty((self.T, self.F + 1)):
            self.__draw_lambda()

        if not self.alpha:
            self.__calculate_alpha()

        # For fitting to convergence:
        self.epsilon = epsilon
        self.max_iter = max_iter


    def __draw_lambda(self):
        """Draw lambda_t for topic t from normal distribution (default values of mu and sigma)"""
        self.lam = np.random.normal(self.mu, math.sqrt(self.sigma), size=self.lam.shape)

    def __calculate_alpha(self):
        """Calculate alpha as exp product of x and lambda transpose matrices"""
        self.alpha = np.exp(np.dot(self.x, self.lam.T))

    def __draw_z(self, data):
        """Draw z from distributions - vector of length n_d for each document, topic """
        #todo parse data into array of 560 by n_d for each
        P = np.empty((self.D, self.m, self.T)) # probability for each topic-mutation pair (z=i, w=j) given alpha and phi

        for d in range(self.D): # for each sample
            for m in range(self.m): # for each category
                for t in range(self.T):  # for each topic
                    P[d, m, t] = self.phi[t, m]*(self.alpha[d, t]+1)/(1+np.sum(self.alpha[d]))
        s = P.sum(axis=2, keepdims=True) #### keep dims?
        P /= s

        # get bin counts of mutations matrix (document by mutations = 560 by 96) and Parse data into array of arrays (560 by n_d for each)

        categories = np.zeros((self.D, self.m))
        B = np.zeros((self.D, self.m))
        _, m = self.phi.shape

        self.z = []
        parsed_data = [None]*self.D
        for id, key in enumerate(data.keys()):
            categories, counts = np.unique(data[key]['sequence'], return_counts=True)
            curr_z = np.zeros(len(data[key]['sequence']))
            for j in range(len(categories)):
                tmp_z = np.random.choice(self.T, size=counts[j], p=P[id, categories[j]])
                curr_z[data[key]['sequence'] == categories[j]] = tmp_z
            self.z.append(curr_z)

            parsed_data[id] = data[key]['sequence']

        # for d in range(self.D):
        #     for m in range(len(data[d])):
        #         ...#todo array of arrays

    def log_likelihood(self, parsed_data, curr_z):
        """Log-likelihood P(z, lambda), using self.z and self.lam"""
        term1 = 1
        term2 = 1
        for d in range(self.D):
            term1 *= gamma(self.alpha)/gamma(self.alpha+len(parsed_data[d]))
            for t in range(self.T):
                term1 *= gamma(self.alpha + curr_z[parsed_data[d]])
        for t in range(self.T):
            for f in range(self.F):
                term2 *= 1/math.sqrt(2*math.pi*self.sigma**2)*math.exp(-self.lam[t][f]**2/(2*self.sigma**2))

        return term1*term2

    def d_log_likelihood(self, t, f, parsed_data, curr_z):
        """Derivative of log-likelihood P(z, lambda), using self.z and self.lam_t_k for topic t and feature f"""
        res = np.sum(np.dot(self.x[f], self.alpha[t]) * (digamma(np.sum(self.alpha, 1)) - digamma(np.sum(self.alpha+parsed_data, 1)) + digamma(self.alpha[t]+parsed_data[t]) - digamma(self.alpha[t]))-self.lam[t][f]/self.sigma**2, 0)
        return res
        # todo fix n_t|d term and double check formula

    def optimize_lambda(self):
        """Receives a lamda and finds new optimal lambda according to bfgs"""
        newlam = optimize.fmin_l_bfgs_b(self.log_likelihood, self.lam, self.d_log_likelihood)[0]
        self.sigma = np.var(newlam, axis=1) #correct axis?
        self.mean = np.mean(newlam, axis=1) #correct axis?
        self.lam = newlam
        self.__calculate_alpha()

    # def new_lambda_no_derivative(self, parsed_data, curr_z):
    #     res = self.log_likelihood()
    #     optimize.newton(func=self.log_likelihood, x0=self.lam, args=(parsed_data, curr_z,))

    def fit(self, data):

        iters = 0
        while iters < self.max_iter:
            self.optimize_lambda()
            self.__draw_z(data)
            iters += 1

        #return estimate

#todo work on below
def load_test_files(sequence_data_filename='simple_data/data_for_michael.json', phi_filename='simple_data/signatures_for_michael.npy'):
    """ Load and parse DNA database """
    with open(sequence_data_filename, 'r') as f:
        data = json.load(f)
    phi = np.load(phi_filename)
    return phi, data


def run_DMR(x, mu=0, sigma=1, epsilon=1e-4, max_iter=1e4):
    phi, data = load_test_files()
    model = DMR(phi, x, mu, sigma, epsilon, max_iter)
    model.fit(data)


    ###### done ######
    # todo: add col of all 1s to x
    # todo parse data into array of 560 by n_d for each
    # todo normalize P by dim 2 (for every d and m, sum should be 1)

    ...  # todo array of arrays


