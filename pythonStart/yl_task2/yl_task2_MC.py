import matplotlib.pyplot as plt
import numpy as np
import initialPR as init
import teskplot as tp
import MC_epsilon_greedy as mc

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
reward = [1,-1,-10,0]
#定义折扣因子gamma和差异
gamma = 0.9
# 初始化状态转移矩阵，奖励矩阵
init.initialPR(P,r,reward,des,forbidden)
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
#开始测试
#题目1
epsilon = [1,0.5]
epsLen = [100, 1000, 10000, 1000000]
for i in range(len(epsilon)):
    for j in range(len(epsLen)):
        Ret,policy,episode = mc.MC_epsilon_greedy(P, r, gamma, epsilon[i], epsLen[j])
        #画路径图
        if j != 3:
            tp.plotTest(episode, False)
        #一百万个点画散点图
        else:
            coor = np.zeros(25 * 5)
            for num in episode:
                no = num[0] * 5 + num[1]
                coor[no] += 1

            # 绘制柱状图
            plt.scatter(np.arange(25 * 5), coor)
            plt.xlabel('状态动作对编号')
            plt.ylabel('出现次数')
            plt.title('状态动作对频率表')
            plt.show()

#题目2
epsilon = [0,0.1,0.2,0.5]
epsLen = 10000
for i in range(len(epsilon)):
    Ret,policy,episode = mc.MC_epsilon_greedy(P, r, gamma, epsilon[i], epsLen)
    V = np.zeros(25)
    for i in range(25):
        V[i] = np.max(Ret[i])
    print("最优状态值：\n", np.round(V.reshape(5, 5), 2))
    print("最优策略：")
    bestPolicy = []
    for j in range(len(policy)):
        a = np.argmax(policy[j])
        bestPolicy.append([j,P[j][a]])
    tp.plotTest(bestPolicy,True)

