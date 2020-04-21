#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt





def PolynomialRegression(bias):
    (countries, features, values) = a1.load_unicef_data()
    targets = values[:,1]
    x = values[:,7:]
    x = a1.normalize_data(x)
    N_TRAIN = 100
    ALL = 195
    x_train = x[0:N_TRAIN,:]
    x_test = x[N_TRAIN:,:]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]
    train_error = {}
    test_error = {}
    for degrees in range(1,7):
        (w, t_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, degrees,0 ,1 ,N_TRAIN,bias)
        (t_est, te_err) = a1.evaluate_regression('polynomial',x_test, w, t_test, degrees, ALL-N_TRAIN, bias)
        print('degree = ',degrees)
        print(t_err)
        train_error[degrees] =   np.sqrt(np.sum(t_err)/100)
        print('sum=', np.sum(t_est,axis=0))
        print('train_error = ',  train_error[degrees])
        test_error[degrees] = np.sqrt(np.sum(te_err)/95)


    for i in range (1,7):
        print(train_error[i])  
    # for i in range (1,7):    
    #     print(test_error[i])
    print(type(train_error))
    plt.rcParams.update({'font.size': 15})

    plt.plot([1,2,3,4,5,6],[train_error[1],train_error[2],train_error[3],train_error[4],train_error[5],train_error[6]])
    plt.plot([1,2,3,4,5,6],[test_error[1],test_error[2],test_error[3],test_error[4],test_error[5],test_error[6]])
    plt.ylabel('RMS')
    plt.legend(['Training error','Testing error'])
    plt.title('Fit with polynomials, no regularization, bias:'+bias)
    plt.xlabel('Polynomial degree')
    plt.show()

PolynomialRegression('yes')
print('_____________________')
PolynomialRegression('no')