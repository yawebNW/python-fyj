import numpy as np

# 打印策略示意图
def print_policy_sketch(policy, shape):
    n_rows, n_cols = shape  # 获取网格的行数和列数
    sketch = [None] * (n_rows * n_cols)  # 初始化策略示意图
    for i in range(n_rows * n_cols):
        dir = policy[i]  # 获取当前状态的动作
        if dir == 1:
            sketch[i] = '^'  # 上移
        elif dir == 2:
            sketch[i] = 'v'  # 下移
        elif dir == 3:
            sketch[i] = '<'  # 左移
        elif dir == 4:
            sketch[i] = '>'  # 右移
        elif dir == 0:
            sketch[i] = 'o'  # 原地不动
    for i in range(0, n_rows * n_cols, n_cols):  # 按行打印策略示意图
        print(sketch[i:i + n_cols])


# 生成轨迹
def generateEpisode(P, r, actPolicy, epiLen):
    episode = []  # 初始化episode列表
    s0 = np.random.randint(25)  # 随机选择初始状态
    a0 = np.random.choice(5, p=actPolicy[0])  # 根据策略选择初始动作
    r0 = r[s0][a0]  # 获取初始奖励
    episode.append([s0, a0, r0, P[s0][a0]])  # 添加到episode
    for i in range(epiLen):  # 对于episode中的每一步
        curState = episode[i][0]  # 当前状态
        curAction = episode[i][1]  # 当前动作
        nextState = P[curState][curAction]  # 下一个状态
        nextAction = np.random.choice(5, p=actPolicy[nextState])  # 根据策略选择下一个动作
        reward = r[nextState][nextAction]  # 获取奖励
        episode.append([nextState, nextAction, reward, P[nextState][nextAction]])  # 添加到episode
    return episode


# 计算均方根误差
def calRMSE(groundTruthVal, vhat):
    sum = 0
    for i in range(len(groundTruthVal)):
        sum += (groundTruthVal[i] - vhat[i]) ** 2
    sum = (sum / len(groundTruthVal)) ** 0.5
    return sum


# 定义使用贝尔曼方程求解状态值向量的函数
def bellman_closed_form(P, r, gamma):
    n = len(P)  # 获取状态数量
    I = np.eye(n)  # 创建单位矩阵
    v = np.linalg.inv(I - gamma * P).dot(r)  # 计算状态值向量
    return v


# 计算真实状态值向量
def getGroundTruth(P, r, gamma):
    # 使用贝尔曼方程求解最优状态值
    Probability = np.zeros((25, 25))  # 状态转移概率矩阵
    rs = np.zeros(25)  # 状态对应平均奖励矩阵
    for i in range(25):
        for j in range(5):
            next_i = P[i][j]
            Probability[i][next_i] += 0.2
            rs[i] += r[i][j] * 0.2

    return bellman_closed_form(Probability, rs, gamma)  # 计算真实状态值
