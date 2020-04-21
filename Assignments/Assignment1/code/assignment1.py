"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'
    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    

def linear_regression(x, t, basis, reg_lambda, degree,  mu, s, N_TRAIN, bias):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function

    phi = design_matrix(basis, bias ,x, N_TRAIN, degree, mu, s )
       
    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        column = (np.size(x, 1))*degree + 1 
        I = np.identity(column)
        tempt_1 = reg_lambda * I
        tempt_2 = np.dot( (np.transpose(phi)),  phi )
        pseudo_inverse = np.linalg.pinv(tempt_1+tempt_2)
        w = np.dot( pseudo_inverse.dot(np.transpose(phi)), t)
    else:
        # no regularization
        pseudo_inverse = np.linalg.pinv(phi)
        w = pseudo_inverse.dot(t)
        
    # Measure root mean squared error on training data.
    train_err = np.power(phi.dot(w)-t,2)
    return (w, train_err)



def design_matrix(basis, bias, x_train, N_TRAIN, degree, mu, s):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        ?????

    Returns:
      phi design matrix
    """
    if basis == 'polynomial':
        if bias=='yes':
            matrix = np.ones((N_TRAIN,1), dtype=np.float64) 
            for i in range (1, degree+1):
               matrix_temp = np.power(x_train, i) 
               matrix = np.concatenate((matrix, matrix_temp), 1) 
            phi = matrix
            return phi
        
        elif bias=='no':
            matrix=x_train
            for i in range (2, degree+1):
                matrix_temp = np.power(x_train, i) 
                matrix = np.concatenate((matrix, matrix_temp), 1) 
            phi = matrix
            return phi
        
    elif basis == 'sigmoid':
        #print("INTO sigma__________________________")
        matrix = np.ones((N_TRAIN,1), dtype=np.float64)  # add bias
        for mu_i in mu:
          matrix_temp = Sigmoidal(x_train, mu_i, s)
          matrix = np.concatenate((matrix, matrix_temp), 1)
        phi = matrix
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(basis,x_test,w, t_test, degree, N_TRAIN, bias,mu=0,s=0):
    """Evaluate linear regression on a dataset.

    Args:
      ?????

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi = design_matrix(basis, bias ,x_test, N_TRAIN, degree,mu,s)
    
    t_est = phi.dot(w)
    err = np.power(t_est-t_test,2)
    return (t_est, err)


def getRMS(error):
    RMS = np.sqrt(np.mean(np.power(error,2)))
    return RMS

def Sigmoidal(x, mu, s):
    ans = 1/(1+np.exp((mu-x)/s))
    return ans