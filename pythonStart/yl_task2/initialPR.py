#根据区域获取奖励
def getReward(i, des, forbidden, reward, toBoundary):
    if toBoundary:return reward[1] #触碰边界时返回reward[1]
    elif i == des:return reward[0] #i就是目标时，奖励为reward[0]
    elif i in forbidden: return reward[2] #i是禁区时，奖励为reward[2]
    return reward[3]#其余返回reward[3]


def initialPR(P,r,reward, des, forbidden):
    # 初始化状态转移概率矩阵 P，和奖励矩阵r
    # 循环处理每个状态
    for i in range(25):
        # 循环处理每个动作
        for j in range(5):
            next_i = i
            toBoundary = False
            if j == 0:
                next_i = i  # 原地不动时就是 i
            # 触碰边界的情况，向上移且是上边界等等
            elif (j == 1 and i < 5) or (j == 2 and i >= 20) or (j == 3 and i % 5 == 0) or (j == 4 and i % 5 == 4):
                next_i = i
                toBoundary = True
            elif j == 1:
                next_i = i - 5  # 不是上边界，移动到状态 i - 5
            elif j == 2:
                next_i = i + 5  # 不是下边界，移动到状态 i + 5
            elif j == 3:
                next_i = i - 1  # 不是左边界，移动到状态 i - 1
            elif j == 4:
                next_i = i + 1  # 不是右边界，移动到状态 i + 1
            P[i][j] = next_i # 记录转移到的状态值
            r[i][j] = getReward(next_i, des, forbidden, reward, toBoundary)

#打印策略示意图
def print_policy_sketch(policy, shape):
    n_rows, n_cols = shape
    sketch = [None] * (n_rows * n_cols)
    for i in range(n_rows * n_cols):
        dir = policy[i]
        if dir == 1:#上移
            sketch[i] = '^'
        elif dir == 2:#下移
            sketch[i] = 'v'
        elif dir == 3:#左移
            sketch[i] = '<'
        elif dir == 4:#右移
            sketch[i] = '>'
        elif dir == 0:#原地不动
            sketch[i] = 'o'
    for i in range(0, n_rows * n_cols, n_cols):
        print(sketch[i:i + n_cols])