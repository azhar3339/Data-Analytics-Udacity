__author__ = 'azhar'
import pandas as pd
import numpy as np
file_name = "/home/azhar/Dropbox/Udacity/Data-Analytics-Udacity/Project1/stroopdata.csv"
stroop_data = pd.read_csv(file_name)

IC = stroop_data['Incongruent']
IC = np.array(IC)

C= stroop_data['Congruent']
C = np.array(C)
# sample_mean = np.mean(IC)
# population_mean = np.mean(C)

def sample_std(X):
    n = X.shape[0]
    mu = np.mean(X)
    print "mean",mu
    temp_diff = X-mu
    temp_sqrd_diff = temp_diff ** 2
    return np.sqrt(sum(temp_sqrd_diff)/(n - 1))



# print "Sample Mean",sample_mean
# print "Population Mean",population_mean
# mean_difference = sample_mean - population_mean
# print "Mean Difference",mean_difference
# difference_from_mean = IC - sample_mean
# # print difference_from_mean[1]
# squared_difference = difference_from_mean**2
# # print squared_difference[1]
# print "n",IC.shape
# sample_std = sum(squared_difference)/23
# print "Sample standard deviation",sample_std
# SEM = sample_std/np.sqrt(24)
# print "Standard Error of Mean",SEM
# t_statistic = mean_difference/SEM
# print "t-statistic",t_statistic

print "Incongruent standard deviation",sample_std(IC)
print "Congruent standard deviation",sample_std(C)

CminusIC = C - IC
print "Difference standard deviation",sample_std(CminusIC)

m = np.mean(CminusIC)
s = sample_std(CminusIC)
t = m/(s/np.sqrt(24))
print "t-statistic",t



