import numpy as np

# 迭代计算贝尔曼最优方程
def optimal_bellman_iteration(P, r, gamma, epsilon):
    """
    该函数使用迭代方法计算贝尔曼最优方程，以找到最优状态值。

    参数：
    P (list of lists of lists)：三维列表，表示状态转移概率矩阵。P[s][a][s_prime] 表示在状态 s 采取动作 a 转移到状态 s_prime 的概率。
    r (numpy.ndarray)：一维数组，表示每个状态的奖励。
    gamma (float)：折扣因子，用于权衡未来奖励与当前奖励的重要性。
    epsilon (float)：收敛阈值，用于判断状态值是否已经收敛。

    返回：
    numpy.ndarray：一维数组，表示最优状态值。
    """
    n = len(r)  # 获取状态数量
    m = len(P[0])  # 获取动作数量
    v = np.zeros(n)  # 初始化当前状态值向量为全零
    v_prime = np.zeros(n)  # 初始化下一次迭代的状态值向量为全零
    while True:
        for s in range(n):  # 遍历每个状态
            max_value = float('-inf')  # 初始化最大价值为负无穷，用于找到每个状态下的最大预期价值
            for a in range(m):  # 遍历每个动作
                sum_value = 0.0  # 初始化总和为0，用于计算采取动作 a 后的预期价值
                for s_prime in range(n):  # 遍历所有可能的下一个状态
                    sum_value += P[s][a][s_prime] * v[s_prime]  # 计算采取动作 a 后转移到状态 s_prime 的概率乘以状态 s_prime 的当前价值，并累加到总和中
                max_value = max(max_value, r[s] + gamma * sum_value)  # 更新最大价值，选择当前动作下的最大预期价值
            v_prime[s] = max_value  # 将状态 s 的最大预期价值赋值给下一次迭代的状态值向量 v_prime
        if np.all(np.abs(v_prime - v) <= epsilon):  # 判断当前状态值向量 v 和下一次迭代的状态值向量 v_prime 是否收敛
            break
        v = v_prime.copy()  # 如果未收敛，将 v_prime 赋值给 v，准备下一次迭代
    return v_prime

# 示例用法
# 定义转移概率矩阵P（这里以简单的3状态3动作示例）
P = [
    [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
]
# 定义奖励向量r
r = np.array([-1, 0, 1])
gamma = 0.9
epsilon = 0.0001

optimal_values = optimal_bellman_iteration(P, r, gamma, epsilon)
print("最优状态值：", optimal_values)

# 根据最优状态值确定最优策略（这里简单打印每个状态的最优动作，实际可能需要更复杂的逻辑）
for s in range(len(optimal_values)):
    best_action = None
    best_value = float('-inf')
    for a in range(len(P[0])):
        sum_value = 0.0
        for s_prime in range(len(optimal_values)):
            sum_value += P[s][a][s_prime] * optimal_values[s_prime]
        if r[s] + gamma * sum_value > best_value:
            best_value = r[s] + gamma * sum_value
            best_action = a
    print(f"状态{s}的最优动作：{best_action}")