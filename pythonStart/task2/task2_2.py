files = 'MovieData.txt'
# 将评分数据写入文件
with open(files, 'w') as f:
    f.write('肖申克的救赎 5 9 \n')
    f.write('肖申克的救赎 3 8 \n')
    f.write('肖申克的救赎 4 7 \n')
    f.write('流浪地球 4 9 \n')
    f.write('上海堡垒 1 2 \n')
    f.write('上海堡垒 3 1 \n')
# 读取文件
with open(files, 'r') as f:
    f = open(files, 'r')
    md = []
    for line in f:
        md.append(line.split(' '))  #读取到列表md中
#计算总评分,可以多次查询,直到输入exit结束
while True:
    search = input('\n请输入需要查询评分的电影:\n')
    if search == 'exit':
        break
    # 计算符合条件的电影的平均评分
    n = 0  # 记录总条数
    allSum = 0  # 记录总分
    for data in md:
        if data[0] == search:
            allSum = allSum + int(data[2])
            n = n + 1
    if n != 0:
        print('电影《%s》的平均评分为：%d' % (search, allSum // n))
    else:
        print('文件中没有《%s》的评分数据' % search)
