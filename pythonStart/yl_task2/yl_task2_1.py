import numpy as np
import initialPR as init
import matplotlib.pyplot as plt

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
epsilon = 0.001

init.initialPR(P,r,reward,des,forbidden)

# 值迭代算法
def value_iteration(P, r, gamma, epsilon, rec=0):
    n = len(P) #n是状态数
    m = len(P[0]) #m是动作数
    #初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    pai = np.zeros(n,dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
    rec_iter = [0]
    # 当价值函数的变化大于阈值 theta 时，继续迭代,简化采用每次价值函数的最大值去比较
    while True:
        delta = 0
        # 遍历每个状态
        for s in range(n):
            #记录最大q值
            q = np.zeros(m)
            #计算所有动作的q值
            for a in range(m):
                reward_s_a = r[s][a] # s状态下采取动作a的奖励，所有概率均为1，所以不用考虑求和
                s_next = P[s][a]# s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
                q[a] = reward_s_a + gamma * V[s_next]
            #更新新值为最大的q
            delta = max(delta,np.max(q) - V[s])
            V[s] = np.max(q)
            #更新策略
            pai[s] = np.argmax(q)
            if s == rec:
                rec_iter.append(V[s])
        # 如果价值函数的变化小于阈值，停止迭代
        if delta < epsilon:
            break
    # 返回最终的价值函数和最优策略
    return V, pai, rec_iter

#策略迭代算法
def policy_iteration(P, r, gamma, epsilon, rec = 0):
    n = len(P) #n是状态数
    m = len(P[0]) #m是动作数
    #初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    pai = np.zeros(n,dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
    rec_iter = [0]
    # 当价值函数的变化大于阈值 theta 时，继续迭代
    while True:
        noPolicyChange_V = V.copy()
        delta1 = 0
        #第二重迭代，持续更新价值函数直到满足阈值
        while True:
            new_V = np.zeros(n)  # 用于存储新的价值函数
            delta2 = 0
        # 策略评估：计算当前策略下的价值函数
            for s in range(n):
                new_V[s] = r[s][pai[s]] + gamma * V[P[s][pai[s]]]
                delta2 = max(delta2, new_V[s] - V[s])
                # 等价于以下代码
                # a = pai[s]  当前策略下s采取的动作，简化为概率为1，所以只有一种，不用求和
                # s_next = P[s][a] s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
                # new_V[s] = r[s][a] + gamma * V[s_next]
            # 如果新的价值函数与旧的价值函数之间的变化小于阈值，停止迭代
            if delta2 < epsilon:
                break
            V = new_V  # 更新价值函数
        for s in range(n):
            q = np.zeros(m)
            for a in range(m):
                reward_s_a = r[s][a]  # s状态下采取动作a的奖励，所有概率均为1，所以不用考虑求和
                s_next = P[s][a]  # s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
                q[a] = reward_s_a + gamma * V[s_next]
            # # 更新策略
            pai[s] = np.argmax(q)
            if s == rec:
                rec_iter.append(V[s])
        # 策略改进：根据当前价值函数改进策略
        delta = max(V[s] - noPolicyChange_V[s] for s in range(n))
        if delta < epsilon:
            break
    # 返回最终的策略和价值函数
    return V, pai, rec_iter

#截断式策略迭代算法
def truncated_policy_iteration(P, r, gamma, epsilon, j, rec = 0):
    n = len(P) #n是状态数
    m = len(P[0]) #m是动作数
    #初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    pai = np.zeros(n,dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
    rec_iter = [0]
    # 当价值函数的变化大于阈值 theta 时，继续迭代
    while True:
        noPolicyChange_V = V.copy()
        delta = 0
        #第二重迭代，截断式，只迭代更新j次
        for x in range(j):
            new_V = np.zeros(n)  # 用于存储新的价值函数
            delta2 = 0
            # 策略评估：计算当前策略下的价值函数
            for s in range(n):
                new_V[s] = r[s][pai[s]] + gamma * V[P[s][pai[s]]]
                delta2 = max(delta2, new_V[s] - V[s])
                # 等价于以下代码
                # a = pai[s]  当前策略下s采取的动作，简化为概率为1，所以只有一种，不用求和
                # s_next = P[s][a] s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
                # new_V[s] = r[s][a] + gamma * V[s_next]
            V = new_V  # 更新价值函数

        for s in range(n):
            q = np.zeros(m)
            for a in range(m):
                reward_s_a = r[s][a]  # s状态下采取动作a的奖励，所有概率均为1，所以不用考虑求和
                s_next = P[s][a]  # s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
                q[a] = reward_s_a + gamma * V[s_next]
            # # 更新策略
            pai[s] = np.argmax(q)
            if s == rec:
                rec_iter.append(V[s])
        # 策略改进：根据当前价值函数改进策略
        delta = max(V[s] - noPolicyChange_V[s] for s in range(n))
        if delta < epsilon:
            break
    # 返回最终的策略和价值函数
    return V, pai, rec_iter

print("值迭代算法")
optimal_values, optimal_policy, rec = value_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
init.print_policy_sketch(optimal_policy,(5,5))
#
# print("策略迭代算法")
# optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
# print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
# print("最优策略：")
# init.print_policy_sketch(optimal_policy,(5,5))
#
jmax=3
# print("截断式策略迭代算法, j = ",jmax)
# optimal_values, optimal_policy = truncated_policy_iteration(P, r, gamma, epsilon,jmax)
# print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
# print("最优策略：")
# init.print_policy_sketch(optimal_policy,(5,5))

# 记录状态17在迭代中的状态值变化，对比三种迭代算法的效果差别
rec = 17
vmax = optimal_values[rec]
v1,p1,rec1 = value_iteration(P, r, gamma, epsilon,rec)
v2,p2,rec2 = policy_iteration(P, r, gamma, epsilon,rec)
v3,p3,rec3 = truncated_policy_iteration(P, r, gamma, epsilon,jmax, rec)
# 绘图
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.axhline(y=vmax,color='purple',linestyle='--',label = "v*")
plt.plot(rec1, c = 'red', label = "值迭代")
plt.plot(rec2, c = 'blue', label = "策略迭代")
plt.plot(rec3, c = 'green', label = "截断式策略迭代")
plt.xlabel("迭代次数")
plt.ylabel("状态值")
plt.legend()
plt.show()

#对比截断式策略迭代算法使用不同值的效果差别
j_array = [1,5,9,56]
v4,p4,rec4 = truncated_policy_iteration(P, r, gamma, epsilon,j_array[0], rec)
v5,p5,rec5 = truncated_policy_iteration(P, r, gamma, epsilon,j_array[1], rec)
v6,p6,rec6 = truncated_policy_iteration(P, r, gamma, epsilon,j_array[2], rec)
v7,p7,rec7 = truncated_policy_iteration(P, r, gamma, epsilon,j_array[3], rec)
# 绘图
plt.axhline(y=vmax,color='purple',linestyle='--',label = "v*")
plt.plot(rec4, c = 'red', label = "j=1")
plt.plot(rec5, c = 'blue', label = "j=5")
plt.plot(rec6, c = 'green', label = "j=9")
plt.plot(rec7, c = 'black', label = "j=56")
plt.xlabel("迭代次数")
plt.ylabel("状态值")
plt.legend()
plt.show()