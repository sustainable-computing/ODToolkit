import numpy as np
import scipy.stats as stats

from odtk.model.superclass import *

class Emission_Model():

    def __init__(self, meta={}):
        self.meta_params = meta

    def learn(self, data):
        """Learn parameters"""

    def get_prob(self, value):
        """Get probability"""
        return 0.0

class Gamma(Emission_Model):

    def __init__(self, meta={}):
        self.meta_params = meta
        
    def learn(self, data):
        self.fit_alpha, self.fit_loc, self.fit_beta = stats.gamma.fit(data)

    def get_prob(self, value):
        p = stats.gamma.pdf(x=value, a=self.fit_alpha, loc=self.fit_loc, scale=self.fit_beta)
        return p

class Guassian(Emission_Model):

    def __init__(self, meta={}):
        self.meta_params = meta
        
    def learn(self, data):
        self.mu, self.std = stats.norm.fit(data)

    def get_prob(self, value):
        p = stats.norm.pdf(x=value, loc=self.mu, scale=self.std)
        return p

class Categorical(Emission_Model):

    def __init__(self, meta={'K':2}):
        self.meta_params = meta
        self.probs = np.zeros(self.meta_params['K'])

    def learn(self, data):
        for i in range(len(data)):
            self.probs[int(data[i])] += 1.
        sum_ = np.sum(self.probs)
        self.probs = self.probs/sum_

    def get_prob(self, value):
        return self.probs[value]

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

    def learn(self, 
              hidden_seq, 
              emission_seq):

       ###################Learning A ############################
        k = len(hidden_seq)-1
        for i in range(k):
            self.A_count[hidden_seq[i]][hidden_seq[i+1]] += 1.

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
                self.B[i].learn(data)
        #################################################

        ###############Learning PI#######################
        self.PI_count[hidden_seq[0]] += 1.

        self.PI = self.PI_count / np.sum(self.PI_count)
        #################################################


    def viterbi_predict(self, emission_seq):

        len_emi = len(emission_seq)
        T1 = np.zeros((self.number_of_hidden_states, len_emi))
        T2 = np.zeros((self.number_of_hidden_states, len_emi))
        
        for i in range(self.number_of_hidden_states):
            T1[i][0] = self.PI[i] * self.B[i].get_prob(emission_seq[0])
            T2[i][0] = 0
        
        for i in range(1, len_emi):
            for j in range(self.number_of_hidden_states):
                T1[j][i] = np.amax(T1[:, i-1]*self.A[:, j]*self.B[j].get_prob(emission_seq[i]))
                T2[j][i] = np.argmax(T1[:, i-1]*self.A[:, j]*self.B[j].get_prob(emission_seq[i]))
        
        z = np.zeros(len_emi, dtype=int)
        x = np.zeros(len_emi, dtype=int)
        
        T = len_emi - 1
        z[T] = np.argmax(T1[:, T])
        x[T] = z[T]
        for i in range(T, 0, -1):
            z[i-1] = T2[z[i]][i]
            x[i-1] = z[i-1]

        return x

    def pf_predict(self, 
                   emission_seq, 
                   number_of_particles, 
                   q, 
                   p=None):

        result = []
        if p is None:
            p = (self.A, self.B)
        s = emission_seq

        particles = np.random.choice(list(range(self.number_of_hidden_states)), size=number_of_particles)
        l = len(s)
        x = []
        for i in range(l):
            w = []
            for j in range(number_of_particles):
                prev = particles[j]
                particles[j] = np.random.choice(list(range(self.number_of_hidden_states)), p=q[prev])
                w_ = p[1][particles[j]].get_prob(s[i])*p[0][prev][particles[j]] / q[prev][particles[j]]
                w.append(w_)
            w = w / np.sum(w)
            new_particles = np.random.choice(particles, size=number_of_particles, p=w)
            particles = new_particles
            
            frequency = np.zeros(self.number_of_hidden_states, dtype=int)
            for parti_ in particles:
                frequency[parti_] += 1
            x.append(np.argmax(frequency))

        return x

