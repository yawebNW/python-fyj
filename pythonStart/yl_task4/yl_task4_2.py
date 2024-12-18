import numpy as np

from pythonStart.yl_task4.DQN import DQN
from pythonStart.yl_task4.initialGridWorld import initialPR

# 初始化5*5网格世界的环境参数
des = 17  # 目标位置
forbidden = [6, 7, 12, 16, 18, 21]  # 墙壁位置
reward = [1, -1, -10, 0]  # 奖励值定义, reward[0]代表目标奖励，reward[1]代表触碰边界的惩罚，reward[2]代表触碰禁区的惩罚，reward[3]代表其他状态的奖励
p,r = initialPR(des, forbidden, reward)
gamma = 0.9

# 初始化策略
policy = np.zeros((25,5)) + 0.2
epiLen1 = 1000
epiLen2 = 100

DQN(p, r, gamma, policy, epiLen1, 500, 100)
DQN(p, r, gamma, policy, epiLen2,500, 100)