#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


def PolynomialRegression(bias,featureNum):
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
    x_trainFeature = x_train[:,featureNum]
    x_testFeature = x_test[:,featureNum]

    (w, t_err) = a1.linear_regression(x_trainFeature, t_train, 'polynomial', 0, 3,0 ,1 ,N_TRAIN,bias)
    (t_est, te_err) = a1.evaluate_regression('polynomial',x_testFeature, w, t_test, 3, ALL-N_TRAIN, bias)
    train_error =  np.sqrt(np.sum(t_err)/100)
    test_error = np.sqrt(np.sum(te_err)/95)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Visulization of feature '+ str(featureNum+8) )
    NumOfPoints = 500
    x_ev1 = np.linspace(np.asscalar(min(x_trainFeature)), np.asscalar(max(x_trainFeature)), num=NumOfPoints)
    x_ev1 = np.array(x_ev1).reshape(NumOfPoints,1)
    phi1 = a1.design_matrix('polynomial', bias, x_ev1, NumOfPoints, 3, 0, 0 )
    y1 = phi1.dot(w)

    x_ev2 = np.linspace(np.asscalar(min(min(x_trainFeature),min(x_testFeature))), np.asscalar(max(max(x_trainFeature) ,max(x_testFeature) )), num=NumOfPoints)
    x_ev2 = np.array(x_ev2).reshape(NumOfPoints,1)
    phi2 = a1.design_matrix('polynomial', bias, x_ev2, NumOfPoints, 3, 0, 0 )
    y2 = phi2.dot(w)

    ax1.plot(x_ev1,y1,'r.-')
    ax1.plot(x_trainFeature,t_train,'bo', color='b')
    ax1.plot(x_testFeature,t_test,'bo',color='g')
    ax2.plot(x_ev2,y2,'r.-')
    ax2.plot(x_trainFeature,t_train,'bo', color='b')
    ax2.plot(x_testFeature,t_test,'bo',color='g')


    
    plt.show()

PolynomialRegression('yes',3)
PolynomialRegression('yes',4)
PolynomialRegression('yes',5)