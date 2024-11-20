import numpy as np

class env:
    def __init__(self,nS,nA):
        self.nS = nS
        self.nA = nA
        self.pai = np.zeros((self.nS, self.nA))

def value_iteration(env, gamma, theta):
    # 初始化价值函数 V，所有状态的初始价值设为0
    V = np.zeros(env.nS)
    # 当价值函数的变化大于阈值 theta 时，继续迭代
    while True:
        delta = 0  # 用于记录价值函数的最大变化量
        # 遍历每个状态
        for s in range(env.nS):
            # 计算当前状态在当前价值函数下的最大价值
            v = V[s]
            V[s] = max([sum([p * (r + gamma * V[s_]) for p, s_, r in env.P[s][a]]) for a in range(env.nA)])
            # 更新最大变化量
            delta = max(delta, abs(v - V[s]))
        # 如果价值函数的变化小于阈值，停止迭代
        if delta < theta:
            break
    # 返回最终的价值函数
    return V


def policy_iteration(env, gamma, theta):
    # 随机初始化策略
    policy = np.random.choice(env.nA, size=env.nS)
    # 初始化价值函数 V
    V = np.zeros(env.nS)
    # 当价值函数的变化大于阈值 theta 时，继续迭代
    while True:
        new_V = np.zeros(env.nS)  # 用于存储新的价值函数
        # 策略评估：计算当前策略下的价值函数
        for s in range(env.nS):
            new_V[s] = sum(
                policy[s] * [sum([p * (r + gamma * V[s_]) for p, s_, r in env.P[s][a]]) for a in range(env.nA)])
        # 如果新的价值函数与旧的价值函数之间的变化小于阈值，停止迭代
        if np.linalg.norm(new_V - V) < theta:
            break
        V = new_V  # 更新价值函数
        # 策略改进：根据当前价值函数改进策略
        for s in range(env.nS):
            policy[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r in env.P[s][a]]) for a in range(env.nA)])
    # 返回最终的策略和价值函数
    return policy, V

def truncated_policy_iteration(env, gamma, theta, truncate):
    # 随机初始化策略
    policy = np.random.choice(env.nA, size=env.nS)
    # 初始化价值函数 V
    V = np.zeros(env.nS)
    # 当价值函数的变化大于阈值 theta 时，继续迭代
    while True:
        new_V = np.zeros(env.nS)  # 用于存储新的价值函数
        # 策略评估：计算当前策略下的价值函数，但只进行 truncate 次迭代
        for j in range(truncate):
            for s in range(env.nS):
                new_V[s] = sum(
                    policy[s] * [sum([p * (r + gamma * V[s_]) for p, s_, r in env.P[s][a]]) for a in range(env.nA)])
        # 如果新的价值函数与旧的价值函数之间的变化小于阈值，停止迭代
        if np.linalg.norm(new_V - V) < theta:
            break
        V = new_V  # 更新价值函数
        # 策略改进：根据当前价值函数改进策略
        for s in range(env.nS):
            policy[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r in env.P[s][a]]) for a in range(env.nA)])
    # 返回最终的策略和价值函数
    return policy, V

# 示例用法：
# env = GridWorldEnv()  # 定义你的环境
# policy, value_function = truncated_policy_iteration(env, gamma=0.9, theta=0.0001, truncate=5)