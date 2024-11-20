import numpy as np
import initialPR as init

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
def value_iteration(P, r, gamma, epsilon):
    n = len(P) #n是状态数
    m = len(P[0]) #m是动作数
    #初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    pai = np.zeros(n,dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
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
        # 如果价值函数的变化小于阈值，停止迭代
        if delta < epsilon:
            break
    # 返回最终的价值函数和最优策略
    return V, pai

value_iteration(P, r, gamma, epsilon)
optimal_values, optimal_policy = value_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
init.print_policy_sketch(optimal_policy,(5,5))

#策略迭代算法
def policy_iteration(P, r, gamma, epsilon):
    n = len(P) #n是状态数
    m = len(P[0]) #m是动作数
    #初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    pai = np.zeros(n,dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
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
        # 策略改进：根据当前价值函数改进策略
        delta1 = max(V[s] - noPolicyChange_V[s] for s in range(n))
        if delta1 < epsilon:
            break
        for s in range(n):
            pai[s] = np.argmax((r[s][a] + gamma * V[P[s][a]]) for a in range(m))
            #等价于以下代码
            # q = np.zeros(m)
            # for a in range(m):
            #     reward_s_a = r[s][a]  # s状态下采取动作a的奖励，所有概率均为1，所以不用考虑求和
            #     s_next = P[s][a]  # s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
            #     q[a] = reward_s_a + gamma * V[s_next]
            # # 更新策略
            # pai[s] = np.argmax(q)
    # 返回最终的策略和价值函数
    return V, pai

policy_iteration(P, r, gamma, epsilon)
optimal_values, optimal_policy = value_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
init.print_policy_sketch(optimal_policy,(5,5))

#截断式策略迭代算法
def truncated_policy_iteration(P, r, gamma, epsilon, j):
    n = len(P) #n是状态数
    m = len(P[0]) #m是动作数
    #初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    pai = np.zeros(n,dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
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
        # 策略改进：根据当前价值函数改进策略
        delta = max(V[s] - noPolicyChange_V[s] for s in range(n))
        if delta < epsilon:
            break
        for s in range(n):
            pai[s] = np.argmax((r[s][a] + gamma * V[P[s][a]]) for a in range(m))
            #等价于以下代码
            # q = np.zeros(m)
            # for a in range(m):
            #     reward_s_a = r[s][a]  # s状态下采取动作a的奖励，所有概率均为1，所以不用考虑求和
            #     s_next = P[s][a]  # s状态下采取动作a后转移到s_next，所有概率均为1，所以不用考虑求和
            #     q[a] = reward_s_a + gamma * V[s_next]
            # # 更新策略
            # pai[s] = np.argmax(q)
    # 返回最终的策略和价值函数
    return V, pai

truncated_policy_iteration(P, r, gamma, epsilon,3)
optimal_values, optimal_policy = value_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
init.print_policy_sketch(optimal_policy,(5,5))
