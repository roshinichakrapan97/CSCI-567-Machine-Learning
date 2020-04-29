import numpy as np

from util import accuracy
from hmm import HMM
import math

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
    
    model = None
    ########################################################

    tags_len = len(tags)
    twod_matrix = np.zeros([tags_len, tags_len])
    phi = np.zeros(tags_len)
    
    observe_map = {}
    state_map = {}
    zz = []
    xi = 0
    
    while xi < tags_len:
        state_map[tags[xi]] = xi
        zz.append([])
        xi += 1
    zz = np.array(zz)
    
    for i in range(len(train_data)):
        data = train_data[i]
        before_tag = data.tags[0]
        phi[state_map[before_tag]] += 1
        present_word = data.words[0]
        
        if present_word not in observe_map:
            observe_map[present_word] = zz.shape[1]
            shape_zz = zz.shape[1]
            zz = np.insert(zz, shape_zz,0,1)
        
        zz[state_map[before_tag]][observe_map[present_word]] += 1
        
        cur_tag_len = len(data.tags)
        
        i = 1
        while i < cur_tag_len:
            present_tag = data.tags[i]
            twod_matrix[state_map[before_tag]][state_map[present_tag]] += 1
            present_word = data.words[i]
            
            if present_word not in observe_map:
                observe_map[present_word] = zz.shape[1]
                shape_zz = zz.shape[1]
                zz = np.insert(zz, shape_zz,0,1)
                
            zz[state_map[present_tag]][observe_map[present_word]] += 1
            before_tag = present_tag
            i+=1
            
    twod_matrix /= twod_matrix.sum(axis = 1)[:, None]
    zz /= zz.sum(axis=1)[:, None]
    phi /= phi.sum()
    
    model = HMM(phi, twod_matrix, zz, observe_map, state_map)
    
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
    tagging = []
    ###################################################
    # Edit here
    for i in range(len(test_data)):
        data = test_data[i]
        for j in range(len(data.words)):
            present_word = data.words[j]
            if present_word not in model.obs_dict:
                shape_B = model.B.shape[1]
                model.obs_dict[present_word] = shape_B
                model.B = np.insert(model.B, shape_B, math.pow(10,-6), 1)    
        tagging.append(model.viterbi(data.words))
     ###################################################
    return tagging
