import numpy as np

from pythonStart.yl_task3.teskplot import plotTest, plotError


def generateEpisode(P, r, actPolicy, epiLen):
    #epi起点是随机生成的
    episode = []
    s0 = 0
    a0 = np.random.choice(5, p = actPolicy[0])
    r0 = r[s0][a0]
    episode.append([s0, a0, r0])
    for i in range(epiLen):
        curState = episode[i][0]
        curAction = episode[i][1]
        nextState = P[curState][curAction]
        nextAction = np.random.choice(5,p = actPolicy[nextState])
        reward = r[nextState][nextAction]
        episode.append([nextState,nextAction, reward])
    return episode


def Q_learning(P, r, actPolicy, truth, epsilon=1, alpha=0.1, epiLen=100000, gamma=0.9, obs=0):
    print('开始Q_learning')
    print('开始生成episode')
    # 初始化episode
    episode = generateEpisode(P,r ,actPolicy,epsilon, epiLen)
    print('episode已生成')
    # 绘制轨迹图
    print('开始绘制轨迹图')
    plotTest(episode)
    print('轨迹图绘制完成')
    # 初始化动作值为0
    q = np.zeros(P.shape)
    bestPolicy = np.zeros(actPolicy.shape)
    obsStateValue = [0]
    # 根据episode进行迭代处理
    for t in range(epiLen - 1):
        s_t = episode[t][0]
        a_t = episode[t][1]
        r_t1 = episode[t][2]
        s_t1 = episode[t+1][0]
        #更新（st，at）对应的q值
        q[s_t][a_t] = q[s_t][a_t] - alpha * (q[s_t][a_t] - (r_t1 + gamma * np.max(q[s_t1])))
        #更新最优策略
        bestPolicy[s_t] = 0
        bestPolicy[s_t][np.argmax(q[s_t])] = 1
        #记录迭代时状态值数据
        if s_t == obs:
           obsStateValue.append(np.max(q[obs]))

    bestQ = np.zeros(len(q))
    for i in range(len(q)):
        bestQ[i] = np.max(q[i])
    plotError(np.array(obsStateValue), truth, obs)
    return bestPolicy,bestQ
