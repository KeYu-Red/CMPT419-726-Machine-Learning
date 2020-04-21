#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)
N_TRAIN = 100
ALL = 195
reg_lambda = [0,0.01,0.1,1,10,100 ,1000 ,10000 ]
x_train_100 = x[:N_TRAIN,:] 
targets_100 = targets[:N_TRAIN,:]
bias = 'yes'
average_validation_set_error =[]

for lambd in reg_lambda:
    sum_validation_set_error=0
    index=0
    for validation in range(0,10):
        x_valid = x_train_100[validation*10:(validation+1)*10 , : ] 
        t_valid = targets_100[validation*10:(validation+1)*10 , : ]
        x_train = np.concatenate((x_train_100[0:validation*10, :], x_train_100[(validation+1)*10:N_TRAIN , : ]) , axis=0) 
        t_train = np.concatenate((targets_100[0:validation*10, :], targets_100[(validation+1)*10:N_TRAIN, : ]), axis=0)
        #def            linear_regression(x, t ,basis, reg_lambda, degree,  mu, s, N_TRAIN, bias):
        (w, t_err) = a1.linear_regression(x_train, t_train, 'polynomial', lambd, 2 , 0, 0 , 90, bias)
        (t_est, te_err) = a1.evaluate_regression('polynomial',x_valid, w, t_valid, 2, 10, bias,0,0)
        train_error=   np.sqrt(np.sum(t_err)/100)
        test_error= np.sqrt(np.sum(te_err)/10)
        sum_validation_set_error = sum_validation_set_error+test_error
    average_validation_set_error.append(sum_validation_set_error/10)
print(average_validation_set_error)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Regularized Polynomial Regression')

ax1.plot([0,0.01,0.1,1,10,100 ,1000 ,10000 ],[average_validation_set_error[0], average_validation_set_error[1], average_validation_set_error[2], average_validation_set_error[3], average_validation_set_error[4], average_validation_set_error[5], average_validation_set_error[6],average_validation_set_error[7]])
ax2.plot([1,2,3,4,5,6,7,8],[average_validation_set_error[0], average_validation_set_error[1], average_validation_set_error[2], average_validation_set_error[3], average_validation_set_error[4], average_validation_set_error[5], average_validation_set_error[6],average_validation_set_error[7]])


plt.show()


