import numpy as np
import tensorflow as tf
from pythonStart.yl_task4.Utils import generateEpisode


def DQN(p,r,gamma,policy, epiLen, iterNum, sampleNum):
    episode = generateEpisode(p,r,policy,epiLen)
    for i in range(iterNum):
        samples = np.random.choice(episode, sampleNum, replace=True)

    return