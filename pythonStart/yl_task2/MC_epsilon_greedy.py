import numpy as np

#MC-epsilon贪心策略算法
def MC_epsilon_greedy(P,r,gamma,epsilon,epiLength):
    n = len(P)  # n是状态数
    m = len(P[0])  # m是动作数
    # 初始化策略矩阵pai，初始策略为均匀分布
    policy = np.ones((n,m))/5
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros((n,m,2))
    Returns = np.zeros((n,m))
    episode = []
    for iter in range(2):
        episode = []
        # 随机选择起始状态动作对
        start = (np.random.randint(n),np.random.randint(m))
        episode.append(start)
        #生成给定长度的episode
        for i in range(epiLength):
            s = episode[i][0]
            a = episode[i][1]
            s_next = P[s][a]#s状态下采取动作a会转到状态s_next
            a_next = np.random.choice(m, size = 1, p = policy[s_next])[0]# 利用贪心策略选择s_next采用的动作
            episode.append((s_next,a_next))#添加新的状态动作对
        g = 0
        #逆序计算每个状态动作对的g值
        for i in range(epiLength - 2, -1, -1):
            s_pre = episode[i+1][0]# 前一个状态
            a_pre = episode[i+1][1]# 前一个动作
            g = g + gamma * r[s_pre][a_pre]# 计算g值需要使用前一个状态采取动作的奖励值
            s = episode[i][0]#当前状态
            a = episode[i][1]#当前动作
            #更新当前状态的价值函数，[0]统计总价值，[1]统计采样次数，不直接求平均防止多次乘除法出现误差
            V[s][a][0] += g
            V[s][a][1] += 1
        #求平均价值
        for i in range(n):
            for j in range(m):
                if V[i][j][1] != 0:
                    Returns[i][j] = V[i][j][0] / V[i][j][1]
        #更新策略
        for i in range(n):
            #q值最大的动作索引
            best_policy = np.argmax(Returns[i])
            for j in range(m):
                if j != best_policy:
                    policy[i][j] = epsilon / m
                else: policy[i][j] = 1 - (m-1) * epsilon / m
    return Returns, policy, episode
