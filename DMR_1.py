import numpy as np
from scipy.special import logsumexp
from scipy.special import gammaln as gammaln
from scipy import optimize as optimize
from scipy.special import digamma as digamma
import json
import math
import csv


class DMR:
    """Topic Model with DMR"""
    def __init__(self, phi, x, data, clinical_data, mu=0, sigma=1, epsilon=1e-4, max_iter=1e4):
        # usually T=12, F=0, D=560, m=96

        ################################# Dimensions ###################################

        self.T, self.m = phi.shape # # topics and # mutation categories
        self.D, self.F = x.shape # # samples/documents and # features in metadata

        #################################### Given #####################################
        self.mu = mu  # mean of lambda's dist
        self.sigma = sigma  # variance of lambda's dist

        self.phi = phi # (e from MMM) - signatures - T by m. probabilities
        temp = np.ones((self.D, self.F + 1))
        temp[:,1:] = x
        self.x = temp # metadata - D by F+1 with column of all 1s

        self.data = data
        self.clinical_data = clinical_data

        ################################################################################

        self.alpha = None # weights for Dir dist - D by T
        self.lam = np.empty((self.T, self.F + 1))
        self.__draw_lambda()

        if not self.alpha:
            self.__calculate_alpha()

        self.z = None # np.empty((self.D, self.n_d))?
        self.n_td = np.empty((self.D, self.T)) # D by T: each entry is how many times topic T drawn in sample D
        self.__draw_z()

        # For fitting to convergence:
        self.epsilon = epsilon
        self.max_iter = max_iter


    def __draw_lambda(self):
        """Draw lambda_t for topic t from normal distribution (default values of mu and sigma)"""
        self.lam = np.random.normal(self.mu, math.sqrt(self.sigma), size=self.lam.shape)

    def __calculate_alpha(self):
        """Calculate alpha as exp product of x and lambda transpose matrices"""
        self.alpha = np.exp(np.dot(self.x, self.lam.T))

    def __draw_z(self):
        """Draw z from distributions - vector of length n_d for each document, topic
           Record number of times topics t drawn for each document d in matrix self.n_td (D by T)
        """

        # Form P

        P = np.empty((self.D, self.m, self.T)) # probability for each topic-mutation pair (z=i, w=j) given alpha and phi

        for d in range(self.D): # for each sample
            for m in range(self.m): # for each category
                for t in range(self.T):  # for each topic
                    P[d, m, t] = self.phi[t, m]*(self.alpha[d, t]+1)/(1+np.sum(self.alpha[d]))
        s = P.sum(axis=2, keepdims=True) #### keep dims?
        P /= s

        # Get bin counts of mutations matrix (document by mutations = 560 by 96)
        # and parse data into array of arrays (560 by n_d for each)

        _, m = self.phi.shape
        self.z = []
        parsed_data = [None]*self.D
        for id, key in enumerate(self.clinical_data): #enumerate(self.data.keys()): # For each sample, get categories and counts
            categories, counts = np.unique(self.data[key]['sequence'], return_counts=True)
            curr_z = np.zeros(len(self.data[key]['sequence']))
            for j in range(len(categories)): # For each mutation category draw counts[j] topics for that particular document's counts array
                tmp_z = np.random.choice(np.arange(self.T), size=counts[j], p=P[id, categories[j]]) # draw counts[j] samples for some mutation category j
                curr_z[self.data[key]['sequence'] == categories[j]] = tmp_z
            self.z.append(curr_z) # len of curr_z is n_d and number in each entry ranges from 0-11 (topics)

            parsed_data[id] = self.data[key]['sequence']
            categories_t, counts_t = np.unique(curr_z, return_counts=True)
            if id == 0: print('curr_z: ', [o for o in curr_z])
            for t in range(self.T):
                self.n_td[id][t] = counts_t[t]


    def log_likelihood_lam(self, lam):
        """Log-likelihood P(z, lambda), using self.z and self.lam"""
        term1 = 0
        term2 = 0

        for d in range(self.D):
            term1 += gammaln(logsumexp(np.dot(self.x, lam.T), axis=1))
            term1 -= gammaln(logsumexp(np.dot(self.x, lam.T)+self.n_td, axis=1))
            for t in range(self.T):
                term1 += gammaln(np.dot(self.x, lam.T)[d][t] + self.n_td[d][t])
                term1 -= gammaln(np.dot(self.x, lam.T))[d][t]
        for t in range(self.T):
            for f in range(self.F):
                term2 += math.log(1/math.sqrt(2*math.pi*self.sigma**2))*(-lam[t][f]**2/(2*self.sigma**2))

        return term1+term2

    def d_log_likelihood(self, alph):
        """Derivative of log-likelihood P(z, lambda), using self.z and self.lam_t_k for topic t and feature f
           NOTE: for now hard coded t and f and d as 0, 0, 0
           NOTE: In terms of alpha, use d_log_likelihood_lam for in terms of lambda.
        """
        t = 0
        f = 0
        d = 0

        res = np.sum(np.dot(self.x.T[f], alph.T[t].T), axis=0) * (digamma(np.sum(alph, axis=1)) - digamma(np.sum(alph, axis=1)+np.sum(self.n_td, axis=1)) + digamma(alph[d][t]+self.n_td[d][t]) - digamma(alph[d][t]))-self.lam[t][f]/self.sigma**2
        return res

    def d_log_likelihood_lam(self, lam):
        """Derivative of log-likelihood P(z, lambda), using self.z and self.lam_t_k for topic t and feature f
           NOTE: for now hard coded t and f and d as 0, 0, 0
        """

        t = 0
        f = 0
        d = 0

        res = np.sum(np.dot(self.x.T[f], np.exp(np.dot(self.x, lam.T)).T[t].T), axis=0)\
              * (digamma(np.sum(np.exp(np.dot(self.x, lam.T)), axis=1))
              - digamma(np.sum(np.exp(np.dot(self.x, lam.T)), axis=1)+np.sum(self.n_td, axis=1))
              + digamma(np.exp(np.dot(self.x, lam.T))[d][t]+self.n_td[d][t])
              - digamma(np.exp(np.dot(self.x, lam.T))[d][t]))-lam[t][f]/self.sigma**2
        return res

    def optimize_lambda(self):
        """Receives a lamda and finds new optimal lambda according to bfgs"""

        def ll(lam):
            lam = np.reshape(lam, (self.T, self.F + 1))
            res = self.log_likelihood_lam(lam)
            return res

        def dll(lam):
            lam = np.reshape(lam, (self.T, self.F + 1))
            res = self.d_log_likelihood_lam(lam)
            res = res.reshape((self.T * (self.F + 1)))
            return res

        random_starting_point = np.random.rand(self.lam.shape[0], self.lam.shape[1])
        newlam = optimize.fmin_l_bfgs_b(ll, random_starting_point, dll)[0]
        newlam = newlam.reshape((self.T * (self.F + 1)))
        self.sigma = np.var(newlam, axis=1) #correct axis to sum over?
        self.mu = np.mean(newlam, axis=1) #correct axis to sum over?
        self.lam = newlam
        self.__calculate_alpha()

    def fit(self):
        """Fit data by stochastic EM"""
        iters = 0
        while iters < self.max_iter:
            self.optimize_lambda()
            self.__draw_z()
            iters += 1
            print(self.lam)

def load_test_files(sequence_data_filename='simple_data/data_for_michael.json', phi_filename='simple_data/signatures_for_michael.npy', metadata='clinical_data_only_binary.csv'):
    """ Load and parse DNA database """
    with open(sequence_data_filename, 'r') as f:
        data = json.load(f)
    phi = np.load(phi_filename)

    with open(metadata, 'r') as metadata1:

        csv_reader = list(csv.reader(metadata1, delimiter=','))
        num_samples = sum(1 for row in csv_reader) - 1
        num_features = 4 #len(csv_reader[0])
        x = np.zeros((num_samples, num_features))
        row_idx = 0
        clinical_data = []
        for row in csv_reader:
            if row_idx != 0:
                #want array of cols 3 to 6 (features)
                for feature_idx, val in enumerate(row[3:]):
                    x[row_idx-1][feature_idx] = val
                clinical_data.append(row[0])
            row_idx += 1
        print(f'Processed {row_idx} lines.')
    return phi, x, data, clinical_data


def run_DMR(mu=0, sigma=1, epsilon=1e-4, max_iter=1e4):
    phi, x, data, clinical_data = load_test_files()
    model = DMR(phi, x, data, clinical_data, mu, sigma, epsilon, max_iter)
    model.fit()

if __name__ == "__main__" or __name__ == 'basic_model_MMM':
    run_DMR()
