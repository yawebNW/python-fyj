import numpy as np

# 根据状态获取奖励
def getReward(i, des, forbidden, reward, toBoundary):
    if i == des:
        return reward[0]  # 如果到达目标状态，返回目标奖励
    elif toBoundary:
        return reward[1]  # 如果触碰边界，返回边界惩罚
    elif i in forbidden:
        return reward[2]  # 如果进入禁区，返回禁区惩罚
    return reward[3]  # 其他情况返回默认奖励


# 初始化状态转移概率矩阵 P 和奖励矩阵 r
def initialPR(des,forbidden,reward):
    # 初始化5*5网格世界的环境参数
    P = np.zeros(25 * 5, dtype=int).reshape(25, 5)  # 状态转移概率矩阵
    r = np.zeros(25 * 5, dtype=int).reshape(25, 5)  # 奖励矩阵

    for i in range(25):  # 遍历每个状态
        for j in range(5):  # 遍历每个动作
            next_i = i  # 默认下一个状态为当前状态
            toBoundary = False  # 默认不触碰边界
            # 根据动作调整下一个状态和边界触碰标志
            if j == 0:
                next_i = i  # 原地不动
            elif (j == 1 and i < 5) or (j == 2 and i >= 20) or (j == 3 and i % 5 == 0) or (j == 4 and i % 5 == 4):
                next_i = i  # 触碰边界，状态不变
                toBoundary = True
            elif j == 1:
                next_i = i - 5  # 向上移动
            elif j == 2:
                next_i = i + 5  # 向下移动
            elif j == 3:
                next_i = i - 1  # 向左移动
            elif j == 4:
                next_i = i + 1  # 向右移动
            P[i][j] = next_i  # 更新状态转移概率矩阵
            r[i][j] = getReward(next_i, des, forbidden, reward, toBoundary)  # 更新奖励矩阵

    return P,r