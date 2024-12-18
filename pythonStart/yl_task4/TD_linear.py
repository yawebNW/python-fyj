import numpy as np
import Utils as util  # 导入Utils模块，假设它包含一些辅助函数
import plotUtils as plot  # 导入teskplot模块，用于绘图

# 定义TD_linear函数，用于策略评估
def TD_linear(P, r, policy, phi, w, alpha, gamma, epiLen, groundTruthVal):
    rmse = np.zeros(epiLen)  # 初始化均方根误差数组
    for k in range(epiLen):  # 对于每个episode
        vhat = np.zeros(25)  # 初始化状态值估计数组
        episode = util.generateEpisode(P, r, policy, epiLen)  # 生成episode
        for i in range(epiLen - 1):  # 对于episode中的每一步
            cur_s = episode[i][0]  # 当前状态
            cur_r = episode[i][2]  # 当前奖励
            next_s = episode[i + 1][0]  # 下一个状态
            phi_s = phi(cur_s)  # 当前状态的特征向量
            phi_ns = phi(next_s)  # 下一个状态的特征向量
            vhat_s = phi_s.dot(w)  # 当前状态的估计值
            vhat_ns = phi_ns.dot(w)  # 下一个状态的估计值
            w = w + alpha * (cur_r + gamma * vhat_ns - vhat_s) * phi_s  # 更新权重
        for j in range(25):  # 计算所有状态的估计值
            vhat[j] = phi(j).dot(w)
        rmse[k] = (util.calRMSE(groundTruthVal, vhat))  # 计算并存储均方根误差
    print('最终w值为:', w)
    print('此时状态值为：\n',vhat.reshape(5,5))
    plot.plotRMSE(epiLen, rmse)  # 绘制均方根误差图
    plot.plotPhi3D(vhat.reshape(5,5))
    return w  # 返回最终的权重向量