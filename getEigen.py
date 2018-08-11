#!/usr/bin/env python
import numpy as np

v = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
              [1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0],
              [1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0],
              [1.0,1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0],
              [1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0],
              [1.0,-1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0],
              [1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0],
              [1.0,-1.0,-1.0,1.0,-1.0,1.0,1.0,-1.0]])

for i in range(len(v)):
    v[i] = v[i] / np.sqrt(sum(v[i]*v[i]))

alpha = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]

n = 8

W = np.matrix(np.zeros((n,n)))

for i in range(len(v)):
    W = W + alpha[i]*np.outer(v[i], v[i])

#print(W)
np.set_printoptions(precision=2)
print(np.linalg.eig(W))
