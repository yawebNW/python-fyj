import matplotlib.pyplot as plt
import numpy as np

def plotTest(episode, isPolicy):
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
    if not isPolicy:
        # 定义路径
        path = [gridNoToCoor(p[0]) for p in episode]
        for i in range(len(path) - 1):
            # 绘制箭头
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            length = np.sqrt(dx ** 2 + dy ** 2)
            ax.arrow(path[i][0], path[i][1], dx, dy, head_width=0.1, head_length=0.1, fc='green', ec='green',
                     length_includes_head=True)
        # 显示图形
        plt.show()
    else:
        for i in range(len(episode)):
            # 绘制箭头
            x,y = gridNoToCoor(episode[i][0])
            dx,dy = gridNoToCoor(episode[i][1])
            length = np.sqrt(dx ** 2 + dy ** 2)
            dx = dx - x
            dy = dy - y
            if length != 0:
                ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='green', ec='green',
                     length_includes_head=True)
            else:
                plt.plot((x,y))
        # 显示图形
        plt.show()