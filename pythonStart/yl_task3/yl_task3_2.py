import numpy as np
import initialPR as init

from pythonStart.yl_task3.Q_learning import Q_learning

#复杂情形，5*5网格世界
# 定义转移概率矩阵P
P = np.zeros(25*5,dtype=int).reshape(25,5)
# 定义奖励矩阵r
r = np.zeros(25*5,dtype=int).reshape(25,5)
#定义目标位置
des = 17
#定义墙壁位置
forbidden = [6,7,12,16,18,21]
#定义奖励，第一个是目标奖励，第二个是边界惩罚，第三个是禁区惩罚，第四个是默认值
reward = [1,-1,-1,0]
#定义折扣因子gamma和学习率alpha
gamma = 0.9
alpha = 0.1
#最优状态值
groundTruthVal = [5.8 ,5.6  ,6.2  ,6.5  ,5.8,
                  6.5 ,7.2  ,8.0  ,7.2  ,6.5,
                  7.2 ,8.0  ,10.0 ,8.0  ,7.2,
                  8.0 ,10.0 ,10.0 ,10.0 ,8.0,
                  7.2 ,9.0  ,10.0 ,9.0  ,8.1]

init.initialPR(P,r,reward,des,forbidden)

#初始策略1是全随机，epsilon=1
actPolicy_1 = np.ones((25, 5)) / 5
#策略2，优先向右，epsilon = 0.5
actPolicy_2 = np.ones((25, 5)) / 10
actPolicy_2[:,4] = 0.6
#策略3，优先向右，epsilon = 0.1
actPolicy_3 = np.zeros((25, 5)) + 0.02
actPolicy_3[:,4] = 0.92
#策略4，随机选择优先方向，epsilon = 0.1
actPolicy_4 = np.zeros((25, 5)) + 0.02
for i in range(25):
    randDir = np.random.randint(0, 4)
    actPolicy_4[i, randDir] = 0.92

allPolicy = [actPolicy_1,actPolicy_2,actPolicy_3,actPolicy_4]
bestPolicy = []
bestAct = []
epsilon = 1
obs = 0
for i in range(4):
    bestPolicy_1, bestAct_1 = Q_learning(P, r, allPolicy[i], epsilon=epsilon, obs=obs, truth = groundTruthVal[obs])
    bestPolicy.append(bestPolicy_1)
    bestAct.append(bestAct_1)