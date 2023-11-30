# -*- coding: utf-8 -*-
import numpy as np

N = 5  # 矩阵大小
empty_label = 0  # 点阵状态为空的标签
error = 0  # 错误返回值
label_n = 3  # n个标签，这里初始化设置为3
label_list = [0, 1, 2]
label_weight = [0.5, 0.3, 0.2]  # n个标签的权值列表【int】类型列表，第0个表示 标签为空的权值

import copy


class point:
    def __init__(self, _label: int):
        """
        :param self.label[int] 点类型的标签值
        :param self.happy_rate[float] 点类型的开心值，范围【0-1.0】
        :param _label: 初始化类中的label的参数
        """
        self.label = _label
        if (_label != empty_label):
            self.happy_rate = 1.0
        else:
            self.happy_rate = None

    def point_set_happy_rate(self, _happy_rate: float):
        self.happy_rate = _happy_rate

    def point_free(self):
        """
        :return: 返回被释放点类的信息，失败【点本来为空】返回error【0】
        """
        print("free point info")
        if (self.label != empty_label and self.happy_rate != None):  # 当点类本身不为空时
            # 创建临时点类信息储存被释放的 标签 和 快乐值
            temp_point = point(self.label)
            temp_point.point_set_happy_rate(self.happy_rate)
            # 设为空值
            self.happy_rate = None
            self.label = empty_label  # empty == 0
            # 返回点类信息
            return temp_point
        else:
            # 当点类本身为空时
            return error


class Matrix:
    def __init__(self, _laberrandom, _N: int):
        """
        :param _laberrandom: 已初始化好的标签数组队列
        """
        self.matrix = get_two_dimensional_matrix_class(_N, _N, label_random=_laberrandom)  # 生成n个点信息到matrix中【二维 N*N】
        self.cow = _N
        self.col = _N


# 实现随机生成N*N大小的一维向量，向量包含所有点的标签信息
def random_label(x: int, y: int, _label_list: list, _weight: list):
    """
    :param _label_list: 标签的列表
    :param x: 二维矩阵的行数
    :param y: 二维矩阵的列
    :param _weight:n种标签的n+1个权值列表【包含空。所以是n+1】
    :return: 返回大小为size的一维标签数组
    """
    size = x * y
    random_res = np.random.choice(_label_list, size=size, p=_weight)
    return random_res


def get_two_dimensional_matrix_class(row: int, col: int, label_random):
    """
    :param row: n行
    :param col: n列
    :param label_random:已经随机过的标签数组，数组大小应该是row * col
    :return: 返回构造好的np.array 二维对象矩阵,错误则为0
    """
    if (len(label_random) != row * col):
        print("row/col not adapt label_random list")
        return error

    for j in range(col):
        for i in range(row):
            temp = j * col + i
            temp_c = point(label_random[temp])
            print(temp_c.label, end=" ")
        print("")
    res = np.array([[point(label_random[j * col + i]) for i in range(row)]
                    for j in range(col)]
                   )
    if (res.shape == (row, col)):  # 确保矩阵输出大小和需求一致
        print("\n matrix size is :", res.shape)
        return res
    else:
        return error


def calculate_happy_rate(input_M: Matrix, row: int, col: int, un_happy_list: list, happy_rate_endurance: float):
    """
    :param happy_rate_endurance:最低忍耐度，小于这个忍耐度会被认为有搬家倾向
    :param input_M: Matrix类
    :param row: 行
    :param col: 列
    :param un_happy_list 不开心的点集合，初始传入可以为空
    :return input_M, 输出新的矩阵类，其实就是改变了他的开心值
            un_happy_list 输出不开心点集
    """
    input_M.matrix = Completion_zero(input_M)  # 将矩阵0填充

    for _col in range(col):
        for _row in range(row):
            temp_col = _col + 1
            temp_row = _row + 1
            temp_ha_r = calculate_8_happy_rate(input_M.matrix, temp_col, temp_row)
            if temp_ha_r < happy_rate_endurance and input_M.matrix[temp_col][temp_row].label != 0:
                un_happy_list.append((temp_col - 1, temp_row - 1))
            input_M.matrix[temp_col][temp_row].happy_rate = temp_ha_r
    print(un_happy_list)
    input_M.matrix = input_M.matrix[1:-1, 1:-1]
    print(input_M.matrix.shape)
    return input_M, un_happy_list


def calculate_8_happy_rate(matrix_input, x: int, y: int):
    """
    :param x: 对应点的坐标位置
    :param y:
    :param matrix_input: 类型同 Matrix.matrix,np.array 点阵的合集【已补零】
    :return:计算单个点的快乐率
    """

    label = matrix_input[x][y].label
    if label == empty_label:
        print(f"matrix_input[{x}][{y}] happy rate is :{0}")
        return 0.0
    count_p = 0.0
    for row in range(3):
        for col in range(3):
            x_temp = x - 1
            y_temp = y - 1
            x_temp += col
            y_temp += row
            if matrix_input[x_temp][y_temp].label == label:
                count_p += 1.0
    count_p -= 1
    # print(count_p)
    happy_rate = count_p / 8.0
    matrix_input[x][y].happy_rate = happy_rate
    print(f"matrix_input[{x}][{y}] happy rate is :{happy_rate}")
    return happy_rate


def Completion_zero(input_M: Matrix):
    """
    :param input_M: 输入的矩阵对象
    :return: 将矩阵对象边缘0填充
    """
    zero_fill = point(0)
    add_M = np.pad(input_M.matrix, 1, constant_values=zero_fill)
    return add_M


# 注重代码规范
if __name__ == "__main__":
    # point 类测试
    # test_point = point(1)
    # print(f"test point lebal is {test_point.label}, happy_rate is {test_point.happy_rate}")

    label_r = random_label(N, N, label_list, label_weight)
    # print(label_r[0:25])
    # t_np = get_two_dimensional_matrix_class(5, 5, label_r)
    # temp = t_np[0][1].label
    # print(f"temp is {temp}")
    test_list = list()
    matrix_t = Matrix(label_r, N)
    # calculate_8_happy_rate(matrix_t.matrix, 1, 3) #测试单个点的快乐值
    matrix_res, test_list = calculate_happy_rate(matrix_t, N, N, test_list, 0.375)
