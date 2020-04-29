"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv, eigvals

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    absolute_error = np.abs(np.dot(X,w)-y)
    mean_abs_error = absolute_error.mean()
    return mean_abs_error

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  inverse_dot = inv(np.dot(X.T,X))
  dot_product = np.dot(inverse_dot,X.T)
  w = np.dot(dot_product, y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    thresh = 1e-05
    dot_product = np.dot(X.T,X)
    min_val = min(eigvals(dot_product))
    l_val = np.identity(dot_product.shape[0]) * 0.1
    while thresh > min_val:
        dot_product = dot_product + l_val
        min_val = min(eigvals(dot_product))
    inverse = inv(dot_product)
    inverse_dot = np.dot(inverse,X.T)
    w = np.dot(inverse_dot,y)
    return w
    
        
    
    
    



###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    dot_product = np.dot(X.T,X)
    size = np.identity(dot_product.shape[0])
    l_val = size * lambd
    dot_product =  dot_product + l_val
    inverse = inv(dot_product)
    inverse_dot = np.dot(inverse,X.T)
    w = np.dot(inverse_dot,y)
    return w
    


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################	
    err = float("inf")
    bestlambda = None
    
    for index in range(-19, 20):
        weight = regularized_linear_regression(Xtrain, ytrain, 10 ** index)
        mean_abs_error = mean_absolute_error(weight, Xval, yval)
        
        if err > mean_abs_error:
            bestlambda = 10 ** index
            err = mean_abs_error
            
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    
    map_X = X
    
    for ind in range(2, power + 1):
        map_X = np.append(map_X, np.power(X,ind), axis=1)
        
    return map_X


