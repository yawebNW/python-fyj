import numpy as np

def MC_epsilon_greedy(P,r,gamma,epsilon,epiLength):
    n = len(P)  # n是状态数
    m = len(P[0])  # m是动作数
    # 初始化策略矩阵pai，初始策略0代表原地不动,这里简化为策略只有一种方向，所以只需要一维矩阵
    policy = np.zeros((n,m), dtype=int)
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(n)
    while True:
        # 随机选择起始状态动作对
        start = (np.random.randint(n),np.random.randint(m))
        episode = [start]
        #生成给定长度的episode
        for i in range(epiLength):
            s = episode[i][0]
            a = episode[i][1]
            s_next = P[s][a]
            a_next = np.random.choice(m, p=epsilon_greedy_policy(epsilon, s, V, m))
            episode.append((s_next,a_next))


def epsilon_greedy_policy(epsilon, state, V, n_actions):
    # 计算当前状态下每个动作的价值
    values = np.zeros(n_actions)
    for a in range(n_actions):
        values[a] = np.mean(V[P[state][a]])
    # 选择最佳动作的概率为 1 - epsilon
    best_action_value = np.max(values)
    best_actions = np.where(values == best_action_value)[0]
    policy = np.zeros(n_actions)
    policy[best_actions] = (1 - epsilon) / len(best_actions)
    # 其他动作的概率为 epsilon / (n_actions - |best_actions|)
    other_actions = np.setdiff1d(np.arange(n_actions), best_actions)
    policy[other_actions] = epsilon / len(other_actions)
    return policy