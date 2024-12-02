import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义正方形区域边界
side_length = 30

# 生成样本
def generate_samples(num_samples):
    """
    使用numpy生成指定数量的样本点。
    参数：
    num_samples (int)：要生成的样本数量。
    返回：
    samples (numpy.ndarray)：生成的样本点数组，形状为(num_samples, 2)，每一行表示一个样本点的坐标。
    """
    # 在边长为side_length的正方形区域内，以原点为中心，随机生成num_samples个x1坐标
    x1 = np.random.uniform(-side_length / 2, side_length / 2, num_samples)
    # 同样随机生成num_samples个x2坐标
    x2 = np.random.uniform(-side_length / 2, side_length / 2, num_samples)
    # 将x1和x2坐标组合成二维数组，每一行是一个样本点的坐标
    return np.column_stack((x1, x2))

# 随机梯度下降估计均值
def sgd_estimate_mean(iter_num, samples, initial_w, learning_rates):
    """
    使用随机梯度下降方法估计均值。
    参数：
    num_samples (int)：样本数量。
    initial_w (list)：初始权重值列表。
    learning_rates (list)：学习率列表，对应每次迭代的学习率。
    返回：
    estimates (numpy.ndarray)：每次迭代后的权重估计值数组，形状为(num_samples, 2)。
    """
    # 将初始权重转换为numpy数组
    w = np.array(initial_w)
    estimates = []
    for i in range(iter_num):
        x = samples[i]
        # 计算梯度，根据随机梯度下降公式，梯度是当前权重与样本点的差值除以迭代次数
        gradient = w - x
        # 更新权重，根据学习率和梯度进行更新
        w = w - learning_rates[i] * gradient
        estimates.append(w.copy())
    return np.array(estimates)

# 小批量梯度下降估计均值
def mbgd_estimate_mean(samples, initial_w, learning_rates, m_values):
    """
    使用小批量梯度下降方法估计均值。
    参数：
    num_samples (int)：样本数量。
    initial_w (list)：初始权重值列表。
    learning_rates (list)：学习率列表。
    m_values (list)：小批量的大小值列表。
    返回：
    all_estimates (dict)：以m值为键，对应小批量梯度下降每次迭代后的权重估计值数组为值的字典。
    """
    w = np.array(initial_w)
    iter_num = len(learning_rates)
    all_estimates = {m: [w] for m in m_values}
    for i in range(iter_num):
        x = samples[i]
        for m in m_values:
            if i % m == 0:
                batch = samples[i:i + m]
                # 计算小批量的梯度，先计算小批量内每个样本点与当前权重的差值之和，再除以小批量大小和迭代次数
                gradient = np.sum(w - batch, axis=0) / m
                w = w - learning_rates[i] * gradient
            all_estimates[m].append(w.copy())
    return all_estimates

# 绘制样本点平面分布和随机梯度下降轨迹
def plot_sgd_results(samples, all_estimates, ak):
    """
    绘制样本点平面分布和随机梯度下降轨迹。
    参数：
    samples (numpy.ndarray)：样本点数组。
    estimates (numpy.ndarray)：随机梯度下降每次迭代后的权重估计值数组。
    """
    plt.figure(figsize=(8, 6))
    # 绘制样本点，蓝色散点
    plt.scatter(samples[:, 0], samples[:, 1], c='lightblue', marker='.', label='Samples')
    for i in range(len(all_estimates)):
        # 绘制随机梯度下降轨迹
        plt.plot(all_estimates[i][:, 0], all_estimates[i][:, 1], label=f'SGD(ak={ak[i]})')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('SGD: Samples and Trajectory')
    plt.legend()
    plt.show()

# 绘制迭代次数与估计误差
def plot_error(all_estimates, ak):
    """
    绘制迭代次数与估计误差的关系图。
    参数：
    num_samples (int)：样本数量。
    estimates (numpy.ndarray)：每次迭代后的权重估计值数组。
    """
    true_value = np.array([0, 0])
    for i in range(len(all_estimates)):
        # 计算每次迭代后的估计误差，使用欧几里得距离公式（这里通过numpy的范数函数计算）
        errors = np.linalg.norm(all_estimates[i] - true_value, axis=1)
        plt.plot(range(1, len(errors) + 1), errors, marker='.', label=f'SGD(ak={ak[i]})')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error vs Iteration (SGD)')
    plt.legend()
    plt.show()

# 绘制小批量梯度下降轨迹
def plot_mbgd_results(samples, all_estimates):
    """
    绘制小批量梯度下降轨迹。
    参数：
    samples (numpy.ndarray)：样本点数组。
    all_estimates (dict)：以m值为键，对应小批量梯度下降每次迭代后的权重估计值数组为值的字典。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], c='lightblue', marker='.', label='Samples')
    for m, estimates in all_estimates.items():
        plt.plot([est[0] for est in estimates], [est[1] for est in estimates], label=f'MBGD (m={m})')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('MBGD: Samples and Trajectories')
    plt.legend()
    plt.show()

def plot_mbgd_error(samples, all_estimates):
    true_value = np.array([0, 0])
    for m,estimates in all_estimates.items():
        # 计算每次迭代后的估计误差，使用欧几里得距离公式（这里通过numpy的范数函数计算）
        errors = np.linalg.norm(estimates - true_value, axis=1)
        plt.plot(range(1, len(errors) + 1), errors, label=f'MBGD (m={m})', marker='.')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error vs Iteration (SGD)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 生成400个样本
    num_samples = 400
    samples = generate_samples(num_samples)
    iter_num = 30

    # 随机梯度下降
    initial_w_sgd = [50, 50]
    learning_rates_sgd_1 = [1 / k for k in range(1, iter_num + 1)]
    learning_rates_sgd_2 = np.zeros(iter_num) + 0.1
    learning_rates_sgd_3 = [1.5 / k for k in range(1, iter_num + 1)]
    sgd_estimates = []
    sgd_estimates_1 = sgd_estimate_mean(iter_num, samples, initial_w_sgd, learning_rates_sgd_1)
    sgd_estimates_2 = sgd_estimate_mean(iter_num, samples, initial_w_sgd, learning_rates_sgd_2)
    sgd_estimates_3 = sgd_estimate_mean(iter_num, samples, initial_w_sgd, learning_rates_sgd_3)
    sgd_estimates.append(sgd_estimates_1)
    sgd_estimates.append(sgd_estimates_2)
    sgd_estimates.append(sgd_estimates_3)
    ak = ['1/k','0.1','1.5/k']
    plot_sgd_results(samples, sgd_estimates, ak)
    plot_error(sgd_estimates, ak)

    # 小批量梯度下降
    initial_w_mbgd = [50, 50]
    learning_rates_mbgd = [1 / k for k in range(1, iter_num + 1)]
    m_values = [1, 10, 50, 100]
    mbgd_estimates = mbgd_estimate_mean(samples, initial_w_mbgd, learning_rates_mbgd, m_values)
    plot_mbgd_results(samples, mbgd_estimates)
    plot_mbgd_error(samples, mbgd_estimates)