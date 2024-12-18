import matplotlib.pyplot as plt
import numpy as np

# 将网格编号转换为坐标
def gridNoToCoor(num):
    x = num % 5 + 0.5  # 计算x坐标，加0.5使网格居中
    y = 4.5 - num // 5  # 计算y坐标，减去整数部分并从4.5开始
    return x, y

# 绘制测试图形
def plotTest(episode):
    fig, ax = plt.subplots()  # 创建图形和子图
    ax.set_xlim(0, 5)  # 设置x轴范围
    ax.set_ylim(0, 5)  # 设置y轴范围
    ax.set_xticks(np.arange(0, 5, 1))  # 设置x轴刻度
    ax.set_yticks(np.arange(0, 5, 1))  # 设置y轴刻度
    ax.set_xticklabels([''] * 5)  # 隐藏x轴刻度标签
    ax.set_yticklabels([''] * 5)  # 隐藏y轴刻度标签
    ax.xaxis.set_ticks_position('none')  # 隐藏x轴刻度
    ax.yaxis.set_ticks_position('none')  # 隐藏y轴刻度
    ax.set_aspect('equal')  # 设置坐标轴比例相等
    ax.grid(True, color='black')  # 显示网格

    # 定义障碍物位置并绘制
    obstacles = [(1, 0), (1, 1), (1, 3), (2, 2), (2, 3), (3, 1), (3, 2)]
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[1]), 1, 1, color='orange', fill=True)
        ax.add_patch(rect)

    # 绘制目的地
    dest = plt.Rectangle((2, 1), 1, 1, color='lightblue', fill=True)
    ax.add_patch(dest)

    # 绘制路径
    path = [gridNoToCoor(p[0]) for p in episode]  # 转换路径坐标
    allArrow = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if (x1 != x2 or y1 != y2) and (not allArrow.__contains__((x1, y1, x2, y2))):
            allArrow.append((x1, y1, x2, y2))
            # 绘制箭头
            dx = x2 - x1
            dy = y2 - y1
            ax.arrow(x1, y1, dx, dy, head_width=0.1, head_length=0.1, fc='green', ec='green',
                     length_includes_head=True)
    plt.show()  # 显示图形

# 计算并绘制误差
def plotError(error, truth, obs):
    error = np.abs(error - truth)  # 计算误差
    plt.plot(np.arange(1, len(error) + 1), error, label=f'观察网格为{obs}号')

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.ylim(0, 6.5)  # 设置y轴范围
    plt.xlabel('迭代次数')  # 设置x轴标签
    plt.ylabel('误差值')  # 设置y轴标签
    plt.title('状态值误差迭代图')  # 设置标题
    plt.legend()  # 显示图例
    plt.show()  # 显示图形

# 绘制3D状态值图
def plotPhi3D(stateValue):
    fig = plt.figure()  # 创建图形
    ax = fig.add_subplot(111, projection='3d')  # 添加3D子图

    x = np.linspace(0, 4, 5)  # 生成x轴数据
    y = np.linspace(0, 4, 5)  # 生成y轴数据
    X, Y = np.meshgrid(x, y)  # 创建网格
    Z = stateValue  # 假设stateValue是预先计算好的Z轴数据

    ax.plot_surface(X, Y, Z, cmap='viridis')  # 绘制曲面图
    ax.set_xlabel('X Label')  # 设置x轴标签
    ax.set_ylabel('Y Label')  # 设置y轴标签
    ax.set_zlabel('Z Label')  # 设置z轴标签

    ax.view_init(elev=45, azim=135)  # 设置视角
    plt.show()  # 显示图形

# 绘制均方根误差图
def plotRMSE(len, rmse):
    fig = plt.figure()  # 创建图形
    plt.plot(np.arange(0, len, 1), rmse, label='RMSE')  # 绘制rmse曲线

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.xlabel('episode计数')  # 设置x轴标签
    plt.ylabel('状态值均方根误差')  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.show()  # 显示图形