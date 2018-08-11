#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

n = 100 # number of input units
m = 50 # number of ouput units

listLength = 20 # number of pairs in each list

nReps = 100

alpha = 0.25

stimSimSet = [0.0, 0.25, 0.5, 0.75, 1.0]

accuracy = [0 for i in range(len(stimSimSet))]

for rep in range(nReps):
    W = np.matrix(np.zeros((m,n))) # m x n matrix
    
    stim1 = []
    resp1 = []
    
    # creat study set
    for litem in range(listLength):
        svec = np.sign(rnd.randn(n)) # numpy.random.randn() corresponds to rnorm()
        stim1.append(svec)

        rvec = np.sign(rnd.randn(m))
        resp1.append(rvec)
    
    # study list
    for litem in range(listLength):
        c = stim1[litem]
        o = resp1[litem]
        W = W + alpha*np.outer(o,c) # corresponding o %*% t(c) in R language

    for stimSimI in range(len(stimSimSet)):
        stimSim = stimSimSet[stimSimI]

        # create test stimuli
        stim2 = []
        for litem in range(listLength):
            svec = np.sign(rnd.randn(n))
            mask = rnd.rand(n) < stimSim # numpy.random.rand() corresponds to runif()
            stim2.append(mask*stim1[litem] + (1-mask)*svec)

        tAcc = 0
        for litem in range(listLength):
            c = stim2[litem]
            o = np.dot(W, c)
        
            tAcc = tAcc + cos_sim(np.array(o).ravel(), resp1[litem])
        accuracy[stimSimI] = accuracy[stimSimI] + tAcc/listLength

accuracy = np.array(accuracy) / nReps

#print("stimSimSet:", stimSimSet)
#print("accuracy:", accuracy)
plt.plot(stimSimSet, accuracy, marker="o")
plt.axis([-0.05, 1.05, -0.05, 1.05]) # The margins were set. Otherwise, the graph would not be like R's one.
plt.xlabel("Stimulus-Cue Similarity")
plt.ylabel("Cosine")
plt.show()
