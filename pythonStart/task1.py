for i in range(1,10):
    for j in range(1,i+1):
        print('%d*%d=%d'%(j,i,i*j),end=" ")
    print()

sum = 0
for i in range(1,100):
    if i%2==1: #奇数用加法
        sum+=i
    else:      #偶数用减法
        sum-=i
print("1-2+3-4+5 ... 99 =",sum)

inputStr = input('请输入一行字符串:\n')
wordNum = 0
numNum = 0
spaceNum = 0
otherNum = 0
for c in inputStr:
    if ('a' <= c <= 'z') or ('A' <= c <= 'Z'): #统计英文字母
        wordNum = wordNum + 1
    elif '0' <= c <= '9': #统计数字
        numNum = numNum + 1
    elif c == ' ': #统计空格
        spaceNum = spaceNum + 1
    else: #统计其他字符
        otherNum = otherNum + 1
print('该字符串共有%d个英文字母, %d个数字, %d个空格, %d个其他字符'%(wordNum, numNum, spaceNum, otherNum))