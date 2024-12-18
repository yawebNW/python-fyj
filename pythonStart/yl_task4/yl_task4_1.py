import numpy as np
import Utils as Init  # 导入Utils模块
from pythonStart.yl_task4.TD_linear import TD_linear  # 导入TD_linear函数
from pythonStart.yl_task4.initialGridWorld import initialPR
from pythonStart.yl_task4.plotUtils import plotPhi3D  # 导入plotPhi3D函数

# 设置numpy打印选项
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# 初始化5*5网格世界的环境参数
des = 17  # 目标位置
forbidden = [6, 7, 12, 16, 18, 21]  # 墙壁位置
reward = [1, -1, -1, 0]  # 奖励值定义
gamma = 0.9 # 折扣因子gamma
P,r = initialPR(des, forbidden, reward)

#获取真实状态值
groundTruthVal = Init.getGroundTruth(P, r, gamma)
print('真实状态值为：\n',groundTruthVal.reshape(5, 5))  # 打印状态值矩阵
plotPhi3D(groundTruthVal.reshape(5, 5))  # 绘制状态值的3D图

# 定义策略，所有动作概率相同
Policy = np.zeros((25, 5)) + 0.2

# 定义不同的特征向量函数
def phi1(s):
    x, y = s % 5, s // 5
    x = (x-2.5) * 2 / 5
    y = (y-2.5) * 2 / 5
    return np.array([1, x, y])

def phi2(s):
    x, y = s % 5, s // 5
    x = (x-2.5) * 2 / 5
    y = (y-2.5) * 2 / 5
    return np.array([1, x, y, x ** 2, y ** 2, x * y])

def phi3(s):
    x, y = s % 5, s // 5
    x = (x-2.5) * 2 / 5
    y = (y-2.5) * 2 / 5
    return np.array([1, x, y, x ** 2, y ** 2, x * y, x ** 3, y ** 3, x ** 2 * y, x * y ** 2])

# 初始化权重向量
w1 = np.zeros(3)
w2 = np.zeros(6)
w3 = np.zeros(10)
alpha1 = 0.0005
alpha2 = 0.0005
alpha3 = 0.0005

# 准备所有特征向量和权重的列表
allParam = [[phi1, w1, alpha1, w1], [phi2, w2, alpha2, w2], [phi3, w3, alpha3, w3]]

# 运行TD_linear算法并计算权重
for j in range(len(allParam)):
    w = TD_linear(P, r, Policy, allParam[j][0], allParam[j][1], allParam[j][2], gamma, 500, groundTruthVal)
    allParam[j][3] = w
