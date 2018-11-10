# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:09:44 2018

@author: Joanna Shen
"""

from sklearn import linear_model as lm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
os.chdir('C:\\Users\\Joanna Shen\\Desktop\\Fall 2018\\Computing')

#Question 1 Part a
data=np.loadtxt("e_car_data.csv", delimiter=',', usecols=(0,4,8,9,7),skiprows = 1)

#APRs offered
x=data[:,:4]
print("x[:10]=\n",x[:10])
print("x has length="+str(len(x)))
#We need x to be a 2D array, hence we do data[:,1:2] not data[:,1]

#Accepted or not
y=(data[:,4]>0)
print("y[:10]=",y[:10])
print("Fraction of loans accepted is", '%.3f'% np.mean(y))

#Create an object of class LogisticRegression
logistic=lm.LogisticRegression()

#Use the fit member function to fit a logistic regression
logistic.fit(x,y)

#Output the intercept and coefficient found here
logistic.intercept_
logistic.coef_

ypred=logistic.predict_proba(x)

#A logistic classifier thresholds the prediction probability at 0.5 by default
print("Error probability of logistic classifier(in-sample):", '%.3f'%(1-logistic.score(x,y)))
print("RMSE of logistic prediction of probability is: ", '%.3f'%np.std(y-ypred[:,1]))

#Question 1 Part b
y = data[:,4]
accept_pred = ypred[:,1]
bin1=(accept_pred<0.2)
#for predicted probability range 0 to 0.2
bin1_success = 0
for i in range(len(bin1)):
    if bin1[i] == True and y[i] == 1:
        bin1_success = bin1_success+1 
print(bin1_success)

cnt = 0
for i in range(len(bin1)):
    if (bin1[i] == True):
        cnt += 1

fraction1 = bin1_success/cnt

#for predicted probability range 0.2 to 0.4
bin2=np.logical_and(accept_pred>=0.2, accept_pred<0.4)

bin2_success = 0
for i in range(len(bin2)):
    if bin2[i] == True and y[i] == 1:
        bin2_success = bin2_success+1 
print(bin2_success)


cnt = 0
for i in range(len(bin2)):
    if (bin2[i] == True):
        cnt += 1

fraction2 = bin2_success/cnt

#for predicted probability range 0.4 to 6
bin3=np.logical_and(accept_pred>=0.4, accept_pred<0.6)

bin3_success = 0
for i in range(len(bin2)):
    if bin3[i] == True and y[i] == 1:
        bin3_success = bin3_success+1 
print(bin3_success)

cnt = 0
for i in range(len(bin3)):
    if (bin3[i] == True):
        cnt += 1

fraction3 = bin3_success/cnt
#for predicted probability range 0.6 to 0.8
bin4=np.logical_and(accept_pred>=0.6, accept_pred<0.8)

bin4_success = 0
for i in range(len(bin4)):
    if bin4[i] == True and y[i] == 1:
        bin4_success = bin4_success+1 
print(bin4_success)


cnt = 0
for i in range(len(bin4)):
    if (bin4[i] == True):
        cnt += 1

fraction4 = bin4_success/cnt
#for predicted probability range 0.8 to 1
bin5=np.logical_and(accept_pred>=0.8, accept_pred<1)

bin5_success = 0
for i in range(len(bin5)):
    if bin5[i] == True and y[i] == 1:
        bin5_success = bin5_success+1 
print(bin5_success)


cnt = 0
for i in range(len(bin5)):
    if (bin5[i] == True):
        cnt += 1

fraction5 = bin5_success/cnt



xbar = ('0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1')
ybar = (fraction1, fraction2, fraction3, fraction4, 0)
x_len = np.arange(len(xbar))
plt.bar(x_len, ybar, align='center', alpha=0.5)
plt.xticks(x_pos,xbar)
plt.ylabel('Fraction of Acceptance from Data')
plt.xlabel('Probability of Acceptance from Model')
plt.show()


#Question 1 Part c
newcus1 = np.array([2, 18000, 5, 2.13])
newcus2 = np.array([2, 30000, 5, 2.13])
np.sum(logistic.coef_*newcus1)


y = data[:,4]
accept_pred = ypred[:,1]
n=0
for n in range (0,1):
    bin=np.logical_and(accept_pred>=n, accept_pred<(n+0.2))
    bin_success = 0
    for i in range(len(bin)):
        if bin[i] == True and y[i] == 1:
            bin_success = bin_success+1 
    print(bin_success)
    n = n+0.2
    
    
    
fraction = bin_success/len(bin)





bin1_success = 0
bin2_success = 0
bin3_success = 0
bin4_success = 0
bin5_success = 0
for i in range(len(data)):
    if accept_pred[i]<0.2 and y[i] == 1:
        bin1_success = bin1_success+1 
    elif accept_pred[i]<0.4 and accept_pred[i]>=0.2 and y[i]==1:
        bin2_success = bin2_success+1
    elif accept_pred[i]<0.6 and accept_pred[i]>=0.4 and y[i]==1:
        bin3_success = bin3_success+1
    elif accept_pred[i]<0.8 and accept_pred[i]>=0.6 and y[i]==1:
        bin4_success = bin4_success+1
    elif accept_pred[i]<1 and accept_pred[i]>=0.8 and y[i]==1:
        bin5_success = bin5_success+1







cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cnt5 = 0
for i in range(len(data)):
    if accept_pred[i]<0.2:
        cnt1 += 1 
    elif accept_pred[i]<0.4 and accept_pred[i]>=0.2:
        cnt2 += 1 
    elif accept_pred[i]<0.6 and accept_pred[i]>=0.4:
        cnt3 += 1 
    elif accept_pred[i]<0.8 and accept_pred[i]>=0.6:
        cnt4 += 1 
    elif accept_pred[i]<1 and accept_pred[i]>=0.8:
        cnt5 += 1 
