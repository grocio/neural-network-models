#!/usr/bin/env python3
import numpy as np

# the function calculating cosine similarity
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

stim = [[1,-1,1,-1],
        [1,1,1,1]]

resp = [[1,1,-1,-1],
        [1,-1,-1,1]]

n = 4 # number of input units
m = 4 # number of output units

W = [[0] * n for i in range(m)]

alpha = 0.25

for pair in range(2):
    for i in range(m):
        for j in range(n):
            W[i][j] = W[i][j] + alpha*stim[pair][j]*resp[pair][i]

o = [0 for i in range(m)]
for i in range(m):
    for j in range(n):
        o[i] = o[i] + W[i][j]*stim[1][j]

print(cos_sim(o, resp[1]))
