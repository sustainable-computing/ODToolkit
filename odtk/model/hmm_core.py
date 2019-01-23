import numpy as np
from scipy.stats import norm

from odtk.model.superclass import *


class Emission_Model():

    def __init__(self, meta={}):
        self.meta_params = meta

    def learn(self, data):
        """Learn parameters"""

    def get_prob(self, value):
        """Get probability"""
        return 0.0


class Gaussian(Emission_Model):

    def __init__(self, meta={}):
        self.meta_params = meta

    def learn(self, data):

        self.n_cols = data.shape[1]
        self.params = {}

        for i in range(self.n_cols):
            mu, sigma = norm.fit(data[:, i])
            self.params[i] = {'mu': mu, 'sigma': sigma}

    def get_prob(self, x):
        p = 1.0
        for i in range(self.n_cols):
            p *= norm.pdf(x=x[i], loc=self.params[i]['mu'], scale=self.params[i]['sigma'])
        return p


class HMM_Core():
    def __init__(self,
                 number_of_hidden_states=2,
                 emission_model=Gaussian,
                 emission_params={}):

        self.number_of_hidden_states = number_of_hidden_states

        self.A_count = np.ones((self.number_of_hidden_states, self.number_of_hidden_states))
        self.PI_count = np.ones(self.number_of_hidden_states)

        self.A = np.zeros((self.number_of_hidden_states, self.number_of_hidden_states))
        self.PI = np.zeros(self.number_of_hidden_states)

        self.B = []
        for i in range(self.number_of_hidden_states):
            emission_obj = emission_model(emission_params)
            self.B.append(emission_obj)

        self.__prior = False

    def prior(self, hmm):

        self.A_count = np.copy(hmm.A_count)
        self.B = hmm.B
        self.PI_count = np.copy(hmm.PI_count)
        self.__prior = True

    def learn(self,
              hidden_seq,
              emission_seq):

        ###################Learning A ############################
        k = len(hidden_seq) - 1
        for i in range(k):
            self.A_count[hidden_seq[i]][hidden_seq[i + 1]] += 1.

        for i in range(self.number_of_hidden_states):
            self.A[i] = self.A_count[i] / np.sum(self.A_count[i])
        ##########################################################

        ##############learning B#########################
        if self.__prior == False:
            for i in range(self.number_of_hidden_states):
                data = []
                for j in range(len(hidden_seq)):
                    if hidden_seq[j] == i:
                        data.append(emission_seq[j])
                self.B[i].learn(np.array(data))
        #################################################

        ###############Learning PI#######################
        self.PI_count[hidden_seq[0]] += 1.

        self.PI = self.PI_count / np.sum(self.PI_count)
        #################################################

    def viterbi_predict(self, emission_seq):

        emission_seq = np.array(emission_seq)
        len_emi = emission_seq.shape[0]
        T1 = np.zeros((self.number_of_hidden_states, len_emi))
        T2 = np.zeros((self.number_of_hidden_states, len_emi))

        for i in range(self.number_of_hidden_states):
            T1[i][0] = self.PI[i] * self.B[i].get_prob(emission_seq[0])
            T2[i][0] = 0

        for i in range(1, len_emi):
            for j in range(self.number_of_hidden_states):
                T1[j][i] = np.amax(T1[:, i - 1] * self.A[:, j] * self.B[j].get_prob(emission_seq[i]))
                T2[j][i] = np.argmax(T1[:, i - 1] * self.A[:, j] * self.B[j].get_prob(emission_seq[i]))

        z = np.zeros(len_emi, dtype=int)
        x = np.zeros(len_emi, dtype=int)

        T = len_emi - 1
        z[T] = np.argmax(T1[:, T])
        x[T] = z[T]
        for i in range(T, 0, -1):
            z[i - 1] = T2[z[i]][i]
            x[i - 1] = z[i - 1]

        return x

    def pf_predict(self,
                   emission_seq,
                   number_of_particles=100):

        particles = np.random.choice(list(range(self.number_of_hidden_states)), size=number_of_particles)
        s = np.array(emission_seq)
        n = s.shape[0]
        x = []
        for i in range(n):
            w = []
            for j in range(number_of_particles):
                particles[j] = np.random.choice(list(range(self.number_of_hidden_states)), p=self.A[particles[j]])
                w.append(self.B[particles[j]].get_prob(s[i]))
            w = w / np.sum(w)
            new_particles = np.random.choice(particles, size=number_of_particles, p=w)
            particles = new_particles

            x.append(round(np.mean(particles)))

        return x
