#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


def PolynomialRegression(bias):
    (countries, features, values) = a1.load_unicef_data()
    targets = values[:,1]
    x = values[:,7:]
    # x = a1.normalize_data(x)
    N_TRAIN = 100
    ALL = 195
    x_train = x[0:N_TRAIN,:]
    x_test = x[N_TRAIN:,:]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]
    train_error = {}
    test_error = {}

    print(x_train)
    for featureNum in range(0,8):
        print('__________',featureNum,'___________________')
        print(x_train[:,featureNum])
    for featureNum in range(0,8):
        x_trainFeature = x_train[:,featureNum]
        x_testFeature = x_test[:,featureNum]
        (w, t_err) = a1.linear_regression(x_trainFeature, t_train, 'polynomial', 0, 3,0 ,1 ,N_TRAIN,bias)
        (t_est, te_err) = a1.evaluate_regression('polynomial',x_testFeature, w, t_test, 3, ALL-N_TRAIN, bias)
        print('featureNum = ',featureNum)
        print(t_err)
        train_error[featureNum] =   np.sqrt(np.sum(t_err)/100)
        print('sum=', np.sum(t_est,axis=0))
        print('train_error = ',  train_error[featureNum])
        test_error[featureNum] = np.sqrt(np.sum(te_err)/95)
        print('train_error = ',  test_error[featureNum])
        print('____________________________')
    x=[8,9,10,11,12,13,14,15]
    x1=[8.35,9.35,10.35,11.35,12.35,13.35,14.35,15.35]
    y_train = [train_error[0],train_error[1],train_error[2],train_error[3],train_error[4],train_error[5],train_error[6],train_error[7]]
    y_test =  [test_error[0],test_error[1],test_error[2],test_error[3],test_error[4],test_error[5],test_error[6],test_error[7]]
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x,y_train,width)
    ax.bar(x1,y_test,0.35)
    plt.ylabel('RMS')
    plt.legend(['Training error','Testing error'])
    plt.title('Fit with polynomials, no regularization, bias:'+bias)
    plt.xlabel('Polynomial degree')
    plt.show()


PolynomialRegression('yes')
# print('_____________________')
PolynomialRegression('no')