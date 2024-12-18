import matplotlib.pyplot as plt
import numpy as np

def plotTest(episode):
    # 创建一个5x5的网格
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_xticklabels([''] * 5)
    ax.set_yticklabels([''] * 5)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_aspect('equal')
    ax.grid(True, color='black')

    # 定义障碍物的位置（这里用橙色表示）
    obstacles = [(1, 0), (1, 1), (1, 3), (2, 2), (2, 3), (3, 1), (3, 2)]
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[1]), 1, 1, color='orange', fill=True)
        ax.add_patch(rect)

    # 绘制目的地
    dest = plt.Rectangle((2, 1), 1, 1, color='lightblue', fill=True)
    ax.add_patch(dest)

    def gridNoToCoor(num):
        x = num % 5 + 0.5
        y = 4.5 - num // 5
        return x, y

    # 绘制路径
    # 定义路径
    path = [gridNoToCoor(p[0]) for p in episode]
    allArrow = []
    for i in range(len(path) - 1):
        x1 = path[i][0]
        y1 = path[i][1]
        x2 = path[i + 1][0]
        y2 = path[i + 1][1]
        if (x1 != x2 or y1 != y2) and (not allArrow.__contains__((x1, y1, x2, y2))):
            allArrow.append((x1, y1, x2, y2))
            # 绘制箭头
            dx = x2 - x1
            dy = y2 - y1
            # length = np.sqrt(dx ** 2 + dy ** 2)
            ax.arrow(x1, y1, dx, dy, head_width=0.1, head_length=0.1, fc='green', ec='green',
                     length_includes_head=True)
    # 显示图形
    plt.show()

def plotError(error, truth, obs):
    error = np.abs(error - truth)
    plt.plot(np.arange(1, len(error) + 1), error, label=f'观察网格为{obs}号')

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决图表中中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号不显示问题

    plt.ylim(0, 6.5)
    plt.xlabel('迭代次数')
    plt.ylabel('误差值')
    plt.title('状态值误差迭代图')
    plt.legend()
    plt.show()