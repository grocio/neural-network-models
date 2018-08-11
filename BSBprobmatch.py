#!/usr/bin/env python
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

v = np.array([[1,1,1,1,-1,-1,-1,-1],
              [1,1,-1,-1,1,1,-1,-1]])

n = 8
maxUpdates = 100
init_s = 1
tol = 1e-8 # tolerance for detecting a difference

alpha = .025

beta = 1
epsilon = 1

nReps = 1000

noiseSet = [0, 0.1, 0.2, 0.4]
startSet = [1/(10-1)*i for i in range(10)]

W = np.matrix(np.zeros((n,n)))

# learning
for i in range(2):
    W = W + alpha*np.outer(v[i],v[i])

# test
sumAcc = []
sumRT = []

for noise in noiseSet:
    tAcc = [0 for i in range(len(startSet))]
    tRT = [0 for i in range(len(startSet))]

    accI = 0 # The index of the values of Accuracy
  
    for startp in startSet:
        for rep in range(nReps):
            u = startp*v[1] + (1-startp)*v[0] # Vector A = v[0], when startp = 0, u = v[0] (vector A)
            u = init_s * u/np.sqrt(sum(u**2))
            u = u + noise*rnd.randn(n) # numpy.random.randn ~ N(0,1). So, to get N(0,sd), sd * randn() is needed
            for t in range(maxUpdates):
                ut = np.copy(u) # Call by reference is used by default
                dot_Wu = np.dot(W,u)
                dot_Wu = np.array(dot_Wu).ravel() # Matrix -> Vector
                u = beta*u + epsilon * dot_Wu 
                u[u > 1] = 1
                u[u < -1] = -1
            
                if all( abs(u-ut) < tol ):
                  break
      
            # is it an A response?
            if all( abs(u-v[0]) < tol ): # Check on the similarity of u and v[0](vector A) based on the differences of each element?
                tAcc[accI] = tAcc[accI] + 1
            # also record response time
            tRT[accI] = tRT[accI] + t
        accI = accI + 1

    # store the results
    sumAcc.append(tAcc)
    sumRT.append(tRT)

meanAcc = np.array(sumAcc) / nReps
meanRT = np.array(sumRT) / nReps

fig, ax = plt.subplots(1,2)

for i in range(len(meanAcc)):
    ax[0].plot(startSet, meanAcc[i], label=str(noiseSet[i]), marker="o")
    ax[0].axis([-0.05, 1.05, -0.05, 1.05])
    ax[0].legend()

for i in range(len(meanRT)):
    ax[1].plot(startSet, meanRT[i], label=str(noiseSet[i]), marker="o")
    ax[1].axis([-0.05, 1.05, -0.05, 20.05])
plt.show()
