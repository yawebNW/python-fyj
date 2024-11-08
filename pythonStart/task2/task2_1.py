import math

class Ima(): #定义父类:图形
    def cal_area(self):#计算面积的方法
        return

class Square(Ima):#正方形对象,继承Ima
    def __init__(self,side):#初始化函数,需要边长
        self.__side = side

    def cal_area(self):#求面积
        return self.__side*self.__side

class Rectangle(Ima):#长方形对象,继承Ima
    def __init__(self,long,width):#初始化,需要长和宽
        self.__long = long
        self.__width = width

    def cal_area(self):#求面积
        return self.__long*self.__width

class Circle(Ima):#圆形对象,继承Ima
    def __init__(self,radius):#初始化,需要半径
        self.__radius = radius

    def cal_area(self):#求面积
        return math.pi * self.__radius*self.__radius

sq = Square(3)
re = Rectangle(2,3)
ci = Circle(3)
print('正方形面积为%f'%sq.cal_area())
print('长方形面积为%f'%re.cal_area())
print('圆形面积为%f'%ci.cal_area())