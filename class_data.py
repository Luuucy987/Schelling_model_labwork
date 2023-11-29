# -*- coding: utf-8 -*-
import numpy as np
N = 50  # 矩阵大小
import copy

class civi:
    def __init__(self):
        self.name = None  # str
        self.label = 0  # 标签
        self.state = 0  # 状态

    def init_by_data(self, _name: str, _label: int, _state: int):
        """
        :except 设定类中的所有变量
        :param _name: 设置的名字
        :param _label: 设定的标签
        :param _state: 设定的状态
        :return:成功设置则返回1，失败则返回0
        """
        self.name = _name
        self.state = _state
        self.label = _label
        if (self.name != None and self.label != None and self.state != None):
            return 1
        else:
            print("init civi error")
            return 0

    def __copy__(self):
        """
        :except:对 对象进行浅拷贝
        :return:该对象（被复制对象）信息
        """
        temp_civi = civi()
        if (temp_civi.init_by_data(self.name, self.label, self.state)):
            return temp_civi
        else:
            print("copy error")
            return 0

    def free(self):
        """
        :except: 释放civi类元素，并返回居民信息
        :return: temp_civi:civi
        """
        temp_civi = civi()
        temp_civi.init_by_data(self.name, self.label, self.state)
        self.name = None  # str
        self.label = None  # 标签
        self.state = None  # 状态设置为空
        return temp_civi  # 返回居民信息

    def print_all(self):
        print(f"name is {self.name}, label is {self.label}, state is {self.state}")


class point:
    def __init__(self, _x: int, _y: int):
        self.x = _x  # 设定x、y
        self.y = _y
        self.Pstate = 0  # 设定初始状态为-->0【未有居民居住】-->1【有公民居住】
        self.Pcivi = civi()

    def set_civi(self, civi_add: civi):
        """
        :except 设置matrix中用户的信息，可用于初始化和入住
        :param civi_add: 添加的用户信息【civi类格式】
        :return 成功则1，不成功则0
        """

        if (self.Pstate == 0):  # 当入住成功
            self.Pcivi = copy.copy(civi_add)  # 调用自定义copy函数对Mcivi进行初始化
            if (self.Pcivi != 0):
                self.Pstate = 1
                print(f"success live in ({self.x},{self.y})")
                self.Pcivi.print_all()
                return 1
        print("set civi in Matrix error")
        return 0

    def free_civi(self):
        """
        :return: 返回被释放的客户信息
        """
        self.Pstate = 0  # 设定该点已搬走【未入住】
        return self.Pcivi.free()  # 调用自定义释放函数

x = np.arange(N)
y = np.arange(N)
X, Y = np.meshgrid(x, y)
status = ["C1", "C2", 'white'] #网格状态，生成三类格子，white为空白网格
prob = [0.4, 0.4, 0.2] #三类格子的生成比例
def init_z():
    """初始化网格
    :return: 返回创建的网格
    """
    Z = np.random.choice(a=status, size=(N**2), p=prob)
    Z.shape = (N, N)
    return Z

def get_cell_happiness(Z, row, col):
    """获取每个单元格的满意程度阈值
    :Z: N*N np.array
    :row: int行, col:int列
    :return: happiness:int
    """
    if not Z.shape == (N, N):
        Z.shape = (N, N)
    if Z[row, col] == "white":
        return np.NaN
    same, count = 0, 0 #same为相同个数，count为有人的个数
    left = 0 if col==0 else col-1
    right = Z.shape[1] if col==Z.shape[1]-1 else col+2
    top = 0 if row==0 else row-1
    bottom = Z.shape[0] if row==Z.shape[0]-1 else row+2

    for i in range(top, bottom):
        for j in range(left, right):
            if (i, j) == (row, col) or Z[i,j] == "white":
                continue
            elif Z[i, j] == Z[row, col]:
                same += 1
                count += 1
            else:
                count += 1
    if not count == 0:
        happiness = same / count
    else:
        happiness = 0
    return happiness

# 注重代码规范
if __name__ == "__main__":
    civi_test = civi()
    civi_test.init_by_data("lucy", 2, 1)
    civi_test.print_all()
    point_t = point(0, 1)
    point_t.set_civi(civi_test)
    Z = init_z()




