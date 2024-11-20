import numpy as np

# 策略迭代算法
def policy_iteration(P, r, gamma, epsilon):
    """
    该函数实现策略迭代算法，用于搜索最优状态值和最优策略。

    参数：
    P (list of lists of lists)：三维列表，表示状态转移概率矩阵。P[s][a][s_prime] 表示在状态 s 采取动作 a 转移到状态 s_prime 的概率。
    r (numpy.ndarray)：二维数组，表示每个状态下采取不同动作的奖励。r[s][a] 表示在状态 s 采取动作 a 的奖励，这里每个动作的奖励可能有多个取值（例如不同的概率对应不同奖励），所以是二维数组。
    gamma (float)：折扣因子，用于权衡未来奖励与当前奖励的重要性。
    epsilon (float)：收敛阈值，用于判断状态值是否已经收敛。

    返回：
    numpy.ndarray：一维数组，表示最优状态值。
    list of lists：二维列表，表示最优策略，其中policy[s][a]表示在状态s采取动作a的概率。
    """
    n = len(r)  # 获取状态数量
    m = len(P[0])  # 获取动作数量

    # 初始化策略，每个动作的概率相等
    policy = np.ones((n, m)) / m

    while True:
        # 策略评估
        v = np.zeros(n)
        while True:
            v_prime = np.zeros(n)
            for s in range(n):
                sum_value = 0.0
                for a in range(m):
                    sum_value += policy[s][a] * (r[s][a] + gamma * sum([P[s][a][s_prime] * v[s_prime] for s_prime in range(n)]))
                v_prime[s] = sum_value
            # 判断状态值是否收敛
            if np.all(np.abs(v_prime - v) <= epsilon):
                break
            v = v_prime.copy()

        # 策略改进
        policy_stable = True
        for s in range(n):
            old_action = np.argmax(policy[s])
            q_values = []
            for a in range(m):
                sum_value = 0.0
                # 计算每个动作的q值
                sum_value += r[s][a] + gamma * sum([P[s][a][s_prime] * v[s_prime] for s_prime in range(n)])
                q_values.append(sum_value)
            new_action = np.argmax(q_values)
            if old_action!= new_action:
                policy_stable = False
            # 更新策略，将最优动作的概率设为1，其他动作概率设为0
            policy[s] = np.eye(m)[new_action]
        if policy_stable:
            break

    return v, policy

#简单情形，1*3网格世界
# 定义转移概率矩阵P
P = [
    [[1, 0, 0], [1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 1, 0], [0, 0, 1], [0, 0, 1]]
]
# 定义奖励向量r（考虑边界负奖励）
r = np.array([[[-1], [0], [1]], [[0], [1], [0]], [[1], [0], [-1]]])
gamma = 0.9
epsilon = 0.1

optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", optimal_values)
print("最优策略：\n")
print(optimal_policy)

#打印策略示意图
def print_policy_sketch(policy, shape):
    n_rows, n_cols = shape
    sketch = [None] * (n_rows * n_cols)
    for i in range(n_rows * n_cols):
        dir = np.argmax(policy[i])
        if dir == 0:#上移
            sketch[i] = '^'
        elif dir == 1:#下移
            sketch[i] = 'v'
        elif dir == 2:#左移
            sketch[i] = '<'
        elif dir == 3:#右移
            sketch[i] = '>'
        elif dir == 4:#原地不动
            sketch[i] = 'o'
    for i in range(0, n_rows * n_cols, n_cols):
        print(sketch[i:i + n_cols])

#根据区域获取奖励
def getReward(i, des, forbidden, reward, toBoundary):
    if toBoundary:return reward[1] #触碰边界时返回reward[1]
    elif i == des:return reward[0] #i就是目标时，奖励为reward[0]
    elif i in forbidden: return reward[2] #i是禁区时，奖励为reward[2]
    return reward[3]#其余返回reward[3]


def initialPR(P,r,reward):
    # 初始化状态转移概率矩阵 P，和奖励矩阵r
    # 循环处理每个状态
    for i in range(25):
        # 循环处理每个动作
        for j in range(5):
            next_i = i
            toBoundary = False
            if j == 4:
                next_i = i  # 原地不动时就是 i
            # 触碰边界的情况，向上移且是上边界等等
            elif (j == 0 and i < 5) or (j == 1 and i >= 20) or (j == 2 and i % 5 == 0) or (j == 3 and i % 5 == 4):
                next_i = i
                toBoundary = True
            elif j == 0:
                next_i = i - 5  # 不是上边界，移动到状态 i - 5
            elif j == 1:
                next_i = i + 5  # 不是下边界，移动到状态 i + 5
            elif j == 2:
                next_i = i - 1  # 不是左边界，移动到状态 i - 1
            elif j == 3:
                next_i = i + 1  # 不是右边界，移动到状态 i + 1
            P[i][j][next_i] = 1
            r[i][j] = getReward(next_i, des, forbidden, reward, toBoundary)

print('\n情况a:')
#复杂情形，5*5网格世界
# 定义转移概率矩阵P
P = np.zeros(25*5*25).reshape(25,5,25)
# 定义奖励矩阵r
r = np.zeros(25*5).reshape(25,5)
#定义目标位置
des = 17
#定义墙壁位置
forbidden = [6,7,12,16,18,21]
#定义奖励，第一个是目标奖励，第二个是边界惩罚，第三个是禁区惩罚，第四个是默认值
reward = [1,-1,-1,0]
#定义折扣因子gamma和差异
gamma = 0.9
epsilon = 0.001

initialPR(P,r,reward)
optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
print_policy_sketch(optimal_policy,(5,5))

#情况2，只修改gamma=0.5
print('\n情况b:')
gamma = 0.5
optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
print_policy_sketch(optimal_policy,(5,5))

#情况3，只修改gamma=0
print('\n情况c:')
gamma = 0
optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
print_policy_sketch(optimal_policy,(5,5))

#情况4，修改禁区奖励为-10
print('\n情况d:')
gamma = 0.9
reward = [1,-1,-10,0]
initialPR(P,r,reward)
optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
print_policy_sketch(optimal_policy,(5,5))

#情况5
print('\n情况e:')
gamma = 0.9
reward = [2,0,0,1]
initialPR(P,r,reward)
optimal_values, optimal_policy = policy_iteration(P, r, gamma, epsilon)
print("最优状态值：\n", np.round(optimal_values.reshape(5,5),2))
print("最优策略：")
print_policy_sketch(optimal_policy,(5,5))