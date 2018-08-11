#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    if np.linalg.norm(v1)*np.linalg.norm(v2) == 0:
        return np.nan
    else:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

n = 100
m = 50

listLength = 20

nReps = 100

alpha = 0.25

lesionPSet = [0.1 * i for i in range(11)]

accuracy = [0 for i in range(len(lesionPSet))]

for rep in range(nReps):
    W = np.matrix(np.zeros((m,n)))

    stim1 = []
    resp1 = []

    # create study set
    for litem in range(listLength):
        svec = np.sign(rnd.randn(n))
        stim1.append(svec)

        rvec = np.sign(rnd.randn(m))
        resp1.append(rvec)

    # study list
    for litem in range(listLength):
        c = stim1[litem]
        o = resp1[litem]
        W = W + alpha*np.outer(o,c)

    for lesionPI in range(len(lesionPSet)):

        lesionP = lesionPSet[lesionPI]
        Wlesion = np.copy(W) # Call by reference by default ?
        mask = rnd.rand(m*n) < lesionP
        mask = mask.reshape((m,n))
        Wlesion[mask] = 0

        tAcc = 0
        for litem in range(listLength):
            c = stim1[litem]
            o = np.dot(Wlesion, c)
            o = np.array(o).ravel()

            tAcc = tAcc + cos_sim(o, resp1[litem])
        accuracy[lesionPI] = accuracy[lesionPI] + tAcc / listLength

accuracy = np.array(accuracy) / nReps

#print("lesionPSet:", lesionPSet)
#print("accuracy:", accuracy)

plt.plot(lesionPSet, accuracy, marker="o")
ax = plt.subplot()
ax.axis([-0.05, 1.05, -0.05, 1.05])
#ax.set_xlim([0.0, 1.0])
#ax.set_ylim([0.0, 1.0])
plt.xlabel("Lesion Probability")
plt.ylabel("Cosine")
plt.show()
