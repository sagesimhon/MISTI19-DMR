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

        self.__organize_data()

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

    def __organize_data(self):

        """For each sample, get bin counts of mutations matrix (document by mutations = 560 by 96), and categories"""
        self.counts = [None]*(self.D-1) #currently hardcoded for 559 samples
        for id, key in enumerate(self.clinical_data):
            if key not in self.data:
                self.clinical_data.remove(key)
                continue
            self.categories, self.counts[id] = np.unique(self.data[key]['sequence'], return_counts=True)

    def __draw_z(self):
        """Draw z from distributions - vector of length n_d for each document, topic
           Record number of times topics t drawn for each document d in matrix self.n_td (D by T)
        """

        ############################### Form P ###############################

        P = np.empty((self.D, self.m, self.T)) # probability for each topic-mutation pair (z=i, w=j) given alpha and phi

        for d in range(self.D): # for each sample
            for m in range(self.m): # for each category
                for t in range(self.T):  # for each topic
                    P[d, m, t] = self.phi[t, m]*(self.alpha[d, t]+1)/(1+np.sum(self.alpha[d]))
        s = P.sum(axis=2, keepdims=True) #### keep dims?
        P /= s

        # parse data into array of arrays (560 by n_d for each) using self.counts/self.categories
        _, m = self.phi.shape
        self.z = []
        parsed_data = [None]*self.D

        for id, key in enumerate(self.clinical_data):
            curr_z = np.zeros(len(self.data[key]['sequence']))
            for j in range(len(self.categories)): # For each mutation category draw counts[j] topics for that particular document's counts array
                tmp_z = np.random.choice(np.arange(self.T), size=self.counts[id][j], p=P[id, self.categories[j]]) # draw counts[j] samples for some mutation category j
                curr_z[self.data[key]['sequence'] == self.categories[j]] = tmp_z
            self.z.append(curr_z) # len of curr_z is n_d and number in each entry ranges from 0-11 (topics)

            parsed_data[id] = self.data[key]['sequence']
            categories_t, counts_t = np.unique(curr_z, return_counts=True)
            # if id == 0: print('curr_z: ', [o for o in curr_z])

            ############################# Update self.n_td accordingly #############################

            for t in range(self.T):
                self.n_td[id][t] = counts_t[t]

        print('draw z')

    def log_likelihood_lam(self, lam):
        """Log-likelihood P(z, lambda), using self.z and self.lam"""
        # term1 = 0
        # term2 = 0
        alpha = np.exp(np.dot(self.x, lam.T))
        # for d in range(self.D):
        #     term1 += gammaln(sum(alpha, axis=1))
        #     term1 -= gammaln(sum(alpha+self.n_td, axis=1))
        #     for t in range(self.T):
        #         term1 += gammaln(alpha[d][t] + self.n_td[d][t])
        #         term1 -= gammaln(alpha)[d][t]
        # for t in range(self.T):
        #     for f in range(self.F):
        #         term2 += math.log(1/math.sqrt(2*math.pi*self.sigma**2))*(-lam[t][f]**2/(2*self.sigma**2))
        #
        # return -sum(term1+term2)
        ll = 0
        for d in range(self.D):
            ll += gammaln(np.sum(alpha[d]))
            ll -= gammaln(np.sum(alpha[d]) + np.sum(self.n_td[d]))
            for t in range(self.T):
                ll += gammaln(alpha[d, t] + self.n_td[d, t])
                ll -= gammaln(alpha[d, t])

        for t in range(self.T):
            for f in range(self.F+1):
                ll -= lam[t, f]**2 / (2*self.sigma**2)
                ll -= np.log(np.sqrt(2*np.pi*self.sigma**2))
        print('ll')
        return -ll

    def d_log_likelihood_lam(self, lam):
        """Derivative of log-likelihood P(z, lambda), using self.z and self.lam_t_k.
           Output: 2-D Array where entry t, f is derivative for topic t and feature f
           NOTE: Used author's implementation for this func"""

        result = np.sum(self.x[:, np.newaxis, :] * np.exp(np.dot(self.x, self.lam.T))[:, :, np.newaxis] \
                        * (digamma(np.sum(np.exp(np.dot(self.x, self.lam.T)), axis=1))[:,np.newaxis,np.newaxis]\
            - digamma(np.sum(self.n_td+np.exp(np.dot(self.x, self.lam.T)), axis=1))[:,np.newaxis,np.newaxis]\
            + digamma(self.n_td+np.exp(np.dot(self.x, self.lam.T)))[:,:,np.newaxis]\
            - digamma(np.exp(np.dot(self.x, self.lam.T)))[:,:,np.newaxis]), axis=0)\
            - lam / (self.sigma ** 2)
        result = -result
        print('dll')
        return result

    def optimize_lambda(self):
        """Receives a lambda and finds new optimal lambda according to bfgs"""

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
        newlam, val, convergence = optimize.fmin_l_bfgs_b(ll, random_starting_point, dll)[0], optimize.fmin_l_bfgs_b(ll, random_starting_point, dll)[1], optimize.fmin_l_bfgs_b(ll, random_starting_point, dll)[2]['warnflag']
        newlam = newlam.reshape((self.T, (self.F + 1)))
        self.sigma = np.var(newlam.T[1]) #todo fix
        self.mu = np.mean(newlam.T[1]) #todo fix
        self.lam = newlam
        self.__calculate_alpha()
        print('optimize lambda')
        return convergence, val

    def fit(self):
        """Fit data by stochastic EM"""
        iters = 0
        f = self.log_likelihood_lam(self.lam)
        counter = 0
        while iters < self.max_iter:
            #c = self.optimize_lambda()[0]
            new_f = self.optimize_lambda()[1]
            self.__draw_z()

            #print('lambda: ', self.lam)
            # if c == 0:
            #     print('CONVERGENCE. Total iterations: ', iters)
            #     print('phi: ', self.phi)
            #     break

            if new_f >= f: # if no improvement in this iteration
                counter += 1
            else:
                counter = 0
            if counter == 10:
                print('No more improvement. Total iterations: ', iters)
                print(new_f)
                break
            print('iters: ', iters)
            f = new_f
            iters += 1


def load_test_files(sequence_data_filename='simple_data/data_for_michael.json', phi_filename='simple_data/signatures_for_michael.npy', metadata='clinical_data.csv'):
    """ Load and parse DNA database """
    with open(sequence_data_filename, 'r') as f:
        data = json.load(f)
    phi = np.load(phi_filename)

    with open(metadata, 'r') as metadata1:

        csv_reader = list(csv.reader(metadata1, delimiter=','))
        num_samples = sum(1 for row in csv_reader) - 1
        num_features = 1 #len(csv_reader[0])
        x = np.zeros((num_samples, num_features))
        row_idx = 0
        clinical_data = []
        for row in csv_reader:
            if row_idx != 0:
                #want array of cols 3 to 6 (features)
                #want array of cols 1 to 1 (age)
                for feature_idx, val in enumerate(row[1:1]):
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
