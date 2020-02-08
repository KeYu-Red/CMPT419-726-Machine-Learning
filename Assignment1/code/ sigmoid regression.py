#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
targets = values[:,1]
x = values[:,7:]
N_TRAIN = 100
ALL = 195
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
featureNum = 3
x_trainFeature = x_train[:,featureNum]
x_testFeature = x_test[:,featureNum]
mu = [100,10000]
s = 2000.0
bias = 'yes'
(w, t_err) = a1.linear_regression(x_trainFeature, t_train, 'sigmoid', 0, 0 ,mu,s ,N_TRAIN,bias)
(t_est, te_err) = a1.evaluate_regression('sigmoid',x_testFeature, w, t_test, 0, ALL-N_TRAIN, bias,mu,s)
train_error=   np.sqrt(np.sum(t_err)/100)
test_error= np.sqrt(np.sum(te_err)/95)
print('train error = ', train_error)
print('test  error = ', test_error)


NumOfPoints=500
x_ev2 = np.linspace(np.asscalar(min(min(x_trainFeature),min(x_testFeature))), np.asscalar(max(max(x_trainFeature) ,max(x_testFeature) )), num=NumOfPoints)
x_ev2 = np.array(x_ev2).reshape(NumOfPoints,1)
phi2 = a1.design_matrix('sigmoid', bias, x_ev2, NumOfPoints, 3, mu, s )
y2 = phi2.dot(w)


plt.plot(x_trainFeature,t_train,'bo', color='b')
plt.plot(x_testFeature,t_test,'bo',color='g')
plt.plot(x_ev2,y2,'r.-')
plt.show()



