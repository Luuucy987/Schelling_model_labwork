# -*- coding: utf-8 -*-
Max_n = 50  # 矩阵大小
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


class matrix:
    def __init__(self, _x: int, _y: int):
        self.x = _x  # 设定x、y
        self.y = _y
        self.Mstate = 0  # 设定初始状态为-->0【未有居民居住】-->1【有公民居住】
        self.Mcivi = civi()

    def set_civi(self, civi_add: civi):
        """
        :except 设置matrix中用户的信息，可用于初始化和入住
        :param civi_add: 添加的用户信息【civi类格式】
        :return 成功则1，不成功则0
        """

        if (self.Mstate == 0):  # 当入住成功
            self.Mcivi = copy.copy(civi_add)  # 调用自定义copy函数对Mcivi进行初始化
            if (self.Mcivi != 0):
                self.Mstate = 1
                print(f"success live in {self.x},{self.y}")
                self.Mcivi.print_all()
                return 1
        print("set civi in Matrix error")
        return 0

    def free_civi(self):
        """
        :return: 返回被释放的客户信息
        """
        self.Mstate = 0  # 设定该点已搬走【未入住】
        return self.Mcivi.free()  # 调用自定义释放函数


# 注重代码规范
if __name__ == "__main__":
    civi_test = civi()
    civi_test.init_by_data("lucy", 2, 1)
    civi_test.print_all()
    matrix_t = matrix(0, 1)
    matrix_t.set_civi(civi_test)
