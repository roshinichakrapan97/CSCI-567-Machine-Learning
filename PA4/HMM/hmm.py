from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        obsd = self.obs_dict[Osequence[0]]
        
        alpha[:,0] = self.pi * self.B[:,obsd]
        
        for j in range(0,L-1,1):
            for i in range(0,S,1):
                dot_prod = np.dot(self.A[:, i].T, alpha[:, j])
                obs_dict_seq = self.obs_dict[Osequence[j+1]]
                alpha[i][j+1] = self.B[i][obs_dict_seq] * dot_prod
                
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        beta[:, L-1] = np.ones(S)
        for j in range(L-1,0,-1):
            for i in range(S):
                obs_dict_trans = (self.A[i, :] * self.B[:, self.obs_dict[Osequence[j]]]).T
                dot_prod = np.dot(obs_dict_trans , beta[:, j])
                beta[i][j-1] = dot_prod
                                  
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        L = len(Osequence)
        prob = np.sum(self.forward(Osequence)[:, L-1])
        
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        den = np.sum(self.forward(Osequence)[:, L-1])
        prob = (self.forward(Osequence) * self.backward(Osequence))/ den
         
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        
        likely_den = np.sum(self.forward(Osequence)[:, L-1])
        for k in range(L-1,0,-1):
            for i in range(0, S, 1):
                for j in range(0, S):
                    l_a = self.forward(Osequence)[i][k-1]
                    l_b = self.backward(Osequence)[j][k]
                    seq_dict = self.obs_dict[Osequence[k]]
                    mat_A = self.A[i][j]
                    mat_B = self.B[j][seq_dict]
                    prob[i][j][k-1] = (mat_A * mat_B * l_a * l_b) / likely_den
                    
        ###################################################
        
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        zed = np.zeros([S,L])
        gam = np.zeros([S,L])
        obsd = self.obs_dict[Osequence[0]]
        gam[:,0] = self.pi * self.B[:, obsd]
        for j in range(0, L-1, 1):
            for i in range(S):
                obs_d = self.obs_dict[Osequence[j+1]]
                gam[i][j+1] = self.B[i][obs_d]
                z = self.A[:, i] * gam[:, j]
                arg_max_index = np.argmax(z)
                zed[i][j+1] = arg_max_index
                gam[i][j+1] *= z[arg_max_index]
                
        indices = np.zeros(L)
        indices[L-1] = np.argmax(gam[:, L-1])
        for x in range(L-1,0, -1):
            ind = int(indices[x])
            indices[x-1] = zed[ind][x]
        
        state_dict_rev = dict(map(reversed, self.state_dict.items()))
        
        for y in range(L):
            ind = int(indices[y])
            path.append(state_dict_rev[ind])
            

        
        ###################################################
        return path
