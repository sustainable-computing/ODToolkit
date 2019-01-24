import numpy as np
import operator

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
        self.empty = False
        
    def learn(self, data):
        if data.shape[0] == 0:
            self.empty = True
            return

        self.n_cols = data.shape[1]
        self.params = {}
        
        for i in range(self.n_cols):
            mu, sigma = norm.fit(data[:, i])
            self.params[i] = {'mu':mu, 'sigma':sigma}

    def get_prob(self, x):
        if self.empty == True:
            return 0.0
        p = 1.0
        for i in range(self.n_cols):
            p=p*norm.pdf(x=x[i], loc=self.params[i]['mu'], scale=self.params[i]['sigma'])

        return p

class HMM_Core():
    def __init__(self, 
                number_of_hidden_states=2, 
                emission_model=Gaussian,
                emission_params={}):

        self.number_of_hidden_states = number_of_hidden_states

        #self.A_count = np.ones((self.number_of_hidden_states, self.number_of_hidden_states))
        #self.PI_count = np.ones(self.number_of_hidden_states)
        
        self.A = np.ones((self.number_of_hidden_states, self.number_of_hidden_states))
        self.PI = np.ones(self.number_of_hidden_states) / self.number_of_hidden_states

        self.B = []
        for i in range(self.number_of_hidden_states):
            emission_obj = emission_model(emission_params)
            self.B.append(emission_obj)
            
        self.__prior = False

    def prior(self, hmm):

        self.A = np.copy(hmm.A)
        self.B = hmm.B
        self.__prior = True

    def learn(self, 
              hidden_seq, 
              emission_seq):

       ###################Learning A ############################
        k = len(hidden_seq)-1
        for i in range(k):
            self.A[hidden_seq[i]][hidden_seq[i+1]] += 1.

        for i in range(self.number_of_hidden_states):
            self.A[i] = self.A[i] / np.sum(self.A[i])
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

    def viterbi_predict (self,emission_seq):
        """Return the best path, given an HMM model and a sequence of emission_seq"""
        emission_seq = np.array(emission_seq)
        #################################initialisation################################ 
        nSamples = emission_seq.shape[0]
        nStates = self.number_of_hidden_states # number of states
        c = np.zeros(nSamples) #scale factors (necessary to prevent underflow)
        viterbi = np.zeros((nStates,nSamples)) # initialise viterbi table
        psi = np.zeros((nStates,nSamples)) # initialise the best path table
        best_path = np.zeros(nSamples, int); # this will be your output
        ##############################################################################
        #######################################initial values for viterbi and best path##################
        for i in range(self.number_of_hidden_states):
            viterbi[i][0] = self.PI[i] * self.B[i].get_prob(emission_seq[0])
        c[0] = 1.0/np.sum(viterbi[:,0])
        viterbi[:,0] = c[0] * viterbi[:,0] # apply the scaling factor
        psi[0] = 0;
        #################################################################################################
        ######################iterations for viterbi and psi for time>0 until T##########################
        for t in range(1,nSamples): # loop through time
            for s in range (0,nStates): # loop through the states @(t-1)
                trans_p = viterbi[:,t-1] * self.A[:,s]
                psi[s,t], viterbi[s,t] = max(enumerate(trans_p), key=operator.itemgetter(1))
                viterbi[s,t] = viterbi[s,t]*self.B[s].get_prob(emission_seq[t])

            c[t] = 1.0/np.sum(viterbi[:,t]) # scaling factor
            viterbi[:,t] = c[t] * viterbi[:,t]
        ##################################################################################################
        ##############################Back-tracking######################################################
        best_path[nSamples-1] =  viterbi[:,nSamples-1].argmax() # last state
        for t in range(nSamples-1,0,-1): # states of (last-1)th to 0th time step
            best_path[t-1] = psi[best_path[t],t]
        #################################################################################################
        return best_path

    def pf_predict(self, 
                   emission_seq, 
                   number_of_particles=50):

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

