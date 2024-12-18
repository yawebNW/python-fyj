import TensorFlow as tf
from pythonStart.yl_task4.Utils import generateEpisode


def DQN(p,r,gamma,policy, epiLen):
    episode = generateEpisode(p,r,policy,epiLen)

    return