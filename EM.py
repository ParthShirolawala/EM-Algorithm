# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 01:45:39 2017

@author: parth
"""

import numpy as np
import pylab as plt

class EM:
    def __init__(self, k=3, eps=0.0001):
        self.k = k  
        self.eps = eps  


    def EMIter(self, xvalue, varrand=1, iteration=100):

        np.random.seed(4)
        n, arraytuple = xvalue.shape
        
        probability = [1. / self.k] * self.k
        
        means = xarr[np.random.choice(n, self.k, False), :]
        variances = np.array([np.eye(arraytuple)] * self.k)
        
        if varrand == 1:
            variances *= np.random.uniform(1, 10)
        else:
            pass

        
        probdistribution = lambda mean, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-xarr.shape[1] / 2.) \
                          * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                                   xarr - mean, np.dot(np.linalg.inv(s), (xarr - mean).T).T))

        
        print("")
        print("Means and covariances for 3 clusters:")
        print("Means:")
        print(means)
        print("Covariance:")
        print(variances)
        
        lglikehoods = []

        
        weightconcentration = np.zeros((n, k))

        
        for i in range(iteration):

            
            for b in range(k):
                weightconcentration[:, b] = probability[b] * probdistribution(means[b], variances[b])

            
            lglikehood = np.sum(np.log(np.sum(weightconcentration, axis=1)))

            lglikehoods.append(lglikehood)

            
            weightconcentration = (weightconcentration.T / np.sum(weightconcentration, axis=1)).T

            
            datapoint = np.sum(weightconcentration, axis=0)

            
            probability = np.zeros(k)
            for a in range(k):
                probability[a] = 1. / n * datapoint[a]

            
            means = np.zeros((k, arraytuple))
            for a in range(k):
                for b in range(n):
                    means[a] += weightconcentration[b, a] * xvalue[b]
                means[a] /= datapoint[a]

            
            if varrand == 1:
                variances = np.zeros((k, arraytuple, arraytuple))
                for a in range(k):
                    for b in range(n):
                        yvalue = np.reshape(xvalue[b] - means[a], (arraytuple, 1))
                        variances[a] += weightconcentration[b, a] * np.dot(yvalue, yvalue.T)
                    variances[a] /= datapoint[a]
            else:
                pass

           
            if len(lglikehoods) < 2:
                continue
            if np.abs(lglikehood - lglikehoods[-2]) < self.eps:
                break

        return probability, means, variances, lglikehoods

    def pltlikelihood(self, lglikehoods):
        
        plt.plot(lglikehoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Number of iterations')
        plt.ylabel('Log likelihood')
        plt.show()


if __name__ == "__main__":


    file = "em_data.txt"
    eps = 0.0001
    
    k = 3

    xarr = np.genfromtxt(file, delimiter=',')
    xarr= xarr.reshape(-1, 1)
    
    print("Implementation of EM algorithm for random covariances")
    varrand = 1
    gmm = EM(k, eps)
    probability, means, variances, lglikehoods = gmm.EMIter(xarr, varrand)
    print("Means and Variances for 3 clusters")
    print("Means:")
    print(means)
    print("Covariance:")
    print(variances)
    print("Iterations before convergence  :",len(lglikehoods))
    print("List of log likelihoods: ",lglikehoods)
    gmm.pltlikelihood(lglikehoods)
    print("")

    
    print("Implementing EM algorithm for covariance as 1")
    varrand = 0
    gmm = EM(k, eps)
    probability, means, variances, lglikehoods = gmm.EMIter(xarr, varrand)
    print("Means and variances for 3 clusters")
    print("Means:")
    print(means)
    print("Covariance:")
    print(variances)
    print("Iterations before convergence :", len(lglikehoods))
    print("List of log likelihoods : ",lglikehoods)
    gmm.pltlikelihood(lglikehoods)

    