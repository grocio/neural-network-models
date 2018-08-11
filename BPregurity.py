#!/usr/bin/env python
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

def logistic_act(x):
  return(1/(1+np.exp(-x)))

nStem = 10
nEnd = 20
nIn = nStem + nEnd
nHid = 15
nOut = 30

nSets = 5
nReg = 4
nIrreg = 1
nPatterns = nSets * (nReg + nIrreg)

inputs = []
outputs = []

nTrain = 6000

eta = 0.1
m = 0.9

for tset in range(nSets):
    stem = rnd.binomial(n = 1, p = 0.5, size = nStem)
    tout = rnd.binomial(n = 1, p = 0.5, size = nOut)
  
    # regular
    for w in range(nReg): # size in the numpy's binomial function means the number of ints unlike R's rbinom
        inputs.append(np.hstack((stem, rnd.binomial(n = 1, p = 0.5, size = nEnd))))
        outputs.append(tout)
  
    # irregular
    inputs.append(np.hstack((stem, rnd.binomial(n = 1, p = 0.5, size = nEnd))))
    outputs.append(rnd.binomial(n = 1, p = 0.5, size = nOut))

Wih = rnd.randn(nHid * nIn)*.01 # Ugly code! Can I directly convert array to matrix? Or should I use ndarray inseted of matrix?
Wih = np.matrix(Wih.reshape((nHid, nIn)))
Who = rnd.randn(nOut * nHid)*.01
Who = np.matrix(Who.reshape((nOut, nHid)))

Bh = np.array([.01] * nHid)
Bo = np.array([.01] * nOut)

dWho = Who*0
dWih = Wih*0

toTrain = range(nPatterns)

error = [0] * nTrain
patterr = np.zeros((nPatterns, nTrain))
patterr[:,:] = np.nan # fill it with nan
patterr = np.matrix(patterr)
patts = np.array([0] * nTrain)

for sweep in range(nTrain): # nTrain is a large number. So, the effect of the randomness should be small.
  
    # which item to train?
    i = rnd.choice(toTrain)
    cue = inputs[i] # each element of inputs[i,] is Int! In the original R code, integer was casted to numeric...
    target = outputs[i]

    ## Cue the network
    net = np.dot(Wih, cue)
    net = np.array(net).ravel() # Matrix -> Vector
    act_hid = logistic_act(net + Bh)
    act_hid = act_hid.transpose()

    net = np.dot(Who, act_hid)
    net = np.array(net).ravel()
    act_out = logistic_act(net + Bo)

    # score up performance
    patterr[i,sweep] = np.sqrt( np.mean((target-act_out)**2) ) # RMSD
    error[sweep] = patterr[i,sweep]
    patts[sweep] = i

    # update hidden-output weights
    d_out = (target - act_out) * act_out * (1 - act_out)


    dWho = eta*np.outer(d_out, act_hid) + m*dWho

    # Backpropagation: update input--hidden weights
    d_hid = np.array(np.dot( Who.transpose(), d_out)).ravel() * act_hid * (1 - act_hid) 
    dWih = eta*np.outer(d_hid, cue) + m*dWih

    # update weights
    Who = Who + dWho 
    Wih = Wih + dWih
    Bo = Bo + eta*d_out
    Bh = Bh + eta*d_hid

regularity = np.array(['reg']*nTrain)
regularity[(patts+1)%5==0] = 'irr'

x_reg = np.array(range(nTrain))[regularity == 'reg']
x_irreg = np.array(range(nTrain))[regularity == 'irr']

error = np.array(error)

y_reg = error[regularity == 'reg']
y_irreg = error[regularity == 'irr']

plt.scatter(x_reg, y_reg, c='blue', label='reg', marker='o', edgecolor='none')
plt.scatter(x_irreg, y_irreg, c='red', label='irreg', marker='o', edgecolor='none')
plt.legend()
plt.show()
