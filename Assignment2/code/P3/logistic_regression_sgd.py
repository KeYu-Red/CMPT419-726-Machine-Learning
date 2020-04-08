#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import random
import assignment2 as a2
from io import StringIO


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
# etas = []

# Load data.
data = np.genfromtxt("data.txt")

# Data matrix, with column of ones at end.
X = data[:, 0:3]

# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]
name=[]
SIZE = len(X)
for eta in etas:
  name.append('eta = '+str(eta))
  # Initialize w.
  w = np.array([0.1, 0, 0])
  # Error values over all iterations.
  e_all = []
  # print("len(X)=",len(X))
  for iter in range(0, max_iter):
    for i in range(0, 100):
      index = random.randint(50,180)
    # Compute output using current w on all data X.
      y = sps.expit(np.dot(X[index], w))
      grad_e = np.multiply((y - t[index]), X[index,:].T)
      w = w - eta*grad_e
    # e is the error, negative log-likelihood (Eqn 4.90)
    y = sps.expit(np.dot(X, w))
    e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
    e_all.append(e)
    #print('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T))

    # Stop iterating if error doesn't change more than tol.
    if iter > 0:
      if np.absolute(e_all[iter] - e_all[iter - 1]) < tol:
        break


  # Plot error over iterations
  TRAIN_FIG = 3
  plt.figure(TRAIN_FIG, figsize=(8.5, 6))
  plt.plot(e_all)

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend(name)
plt.show()
