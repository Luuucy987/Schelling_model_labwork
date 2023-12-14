# -*- coding: utf-8 -*-
import numpy as np
import visualization

N = 10  # 矩阵大小
empty_label = 0  # 点阵状态为空的标签
error = 0  # 错误返回值
label_n = 3  # n个标签，这里初始化设置为3
label_list = [0, 1, 2]
label_weight = [0.5, 0.3, 0.2]  # n个标签的权值列表【int】类型列表，第0个表示 标签为空的权值

import copy


# class_c[n][n]

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
        print(f"free point info:label is {self.label}")
        if (self.label != empty_label and self.happy_rate != None):  # 当点类本身不为空时
            # 创建临时点类信息储存被释放的 标签 和 快乐值
            temp_point = point(self.label)
            temp_point.point_set_happy_rate(self.happy_rate)
            # 设为空值
            self.happy_rate = 0
            self.label = empty_label  # empty == 0
            # 返回点类信息
            return temp_point
        else:
            # 当点类本身为空时
            print("free point is empty ERROR")
            return error


class Matrix:
    def __init__(self, _laberrandom, _N: int):
        """
        :param _laberrandom: 已初始化好的标签数组队列
        """
        self.matrix = get_two_dimensional_matrix_class(_N, _N, label_random=_laberrandom)  # 生成n个点信息到matrix中【二维 N*N】
        self.row = _N  # 行
        self.col = _N  # 列


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
                un_happy_list.append((temp_row - 1, temp_col - 1))
            input_M.matrix[temp_col][temp_row].happy_rate = temp_ha_r
    print(un_happy_list)
    input_M.matrix = input_M.matrix[1:-1, 1:-1]
    print(input_M.matrix.shape)
    return input_M, un_happy_list


def calculate_8_happy_rate(matrix_input, x: int, y: int):
    """
    初始化时调用计算点的快乐值【无视标签为空】
    :param x: 对应点的坐标位置
    :param y:
    :param matrix_input: 类型同 Matrix.matrix,np.array 点阵的合集【已补零】
    :return:计算单个点的快乐率
    """

    label = matrix_input[x][y].label
    if label == empty_label:
        # print(f"matrix_input[{x}][{y}] happy rate is :{0}")
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
    # print(f"matrix_input[{x}][{y}] happy rate is :{happy_rate}")
    return happy_rate


def Completion_zero(input_M: Matrix):
    """
    :param input_M: 输入的矩阵对象
    :return: 将矩阵对象边缘0填充
    """
    zero_fill = point(0)
    add_M = np.pad(input_M.matrix, 1, constant_values=zero_fill)
    return add_M


def updata_8_point_happy_rate(input_mp: np.array, x: int, y: int, ori_tuple: tuple, happy_rate_k: float,
                              un_happy_list: list):
    """
    为了更新 新点四周的开心值
    :param input_mp: 输入的二维点矩阵
    :param x: 要更新点的坐标
    :param y:
    :param ori_tuple:被移动点的坐标信息元组
    :param happy_rate_k:开心值的阈值
    :param un_happy_list:不开心的列表
    :return: 返回更新完成的二维点矩阵，和输入同理
    """
    ori_x, ori_y = ori_tuple[1] + 1, ori_tuple[0] + 1
    print(f"update {x},{y}")
    # 尝试移动到指定点
    input_mp[x][y] = input_mp[ori_x][ori_y].point_free()

    # 更新8临域
    for row in range(3):
        for col in range(3):
            x_temp = x - 1
            y_temp = y - 1
            x_temp += col
            y_temp += row
            temp_happy_rate = calculate_8_happy_rate(input_mp, x_temp, y_temp)

            input_mp[x][y].happy_rate = temp_happy_rate
            # if (x_temp - 1, y_temp - 1) in un_happy_list and temp_happy_rate > happy_rate_k:
            #     un_happy_list.remove((x_temp-1,y_temp-1))
            #     print(f"un_happy remove point {x_temp - 1},{y_temp - 1}")
    un_happy_list.pop(0)
    return un_happy_list, input_mp


def matrix_move_1point(inpute_M: Matrix, tupel_x_y: tuple, happy_rate_k: float, unhappy_list: list, max_k=25):
    """
    :param inpute_M: 输入矩阵类
    :param tupel_x_y: 原【打算搬家节点信息x，y元组】
    :param happy_rate_k:  开心值的阈值
    :param unhappy_list: 不开心的列表
    :param max_k:最大迭代次数【超过找不到就退出】
    :return:
    """
    # 0扩充
    # if type(inpute_M.matrix[tupel_x_y[1],tupel_x_y[0]].happy_rate) != happy_rate_k:
    #     unhappy_list.pop(0)
    #     return unhappy_list, inpute_M.matrix
    if inpute_M.matrix[tupel_x_y[1],tupel_x_y[0]].happy_rate >= happy_rate_k:
        print("suit")
        unhappy_list.pop(0)
        print(f"un happy list :\n {unhappy_list}")
        return unhappy_list, inpute_M.matrix
    matrix_p = Completion_zero(inpute_M)  # 点矩阵
    # visualization.plot_grid_label(matrix_p)  # -------------------------可视化测试
    # visualization.plot_grid_happy_rate(matrix_p, 0.375)
    x = tupel_x_y[1] + 1
    y = tupel_x_y[0] + 1
    # print(f"start move point is[{x},{y}]-------------matrix_move_1point")
    # print(f"ori point info [{matrix_p[x][y].label},{matrix_p[x][y].happy_rate}]-------------matrix_move_1point")
    mod_x = inpute_M.row + 1
    mod_y = inpute_M.col + 1
    ori_label = matrix_p[x][y].label
    max_pro_happyrate = matrix_p[x][y].happy_rate
    outOP_x_y = (tupel_x_y[0], tupel_x_y[1])
    while max_k > 0:
        if (x + 1) % mod_x == 0 and x != 0:
            if (y + 1) % mod_y == 0 and y != 0:
                x = (x + 1) % mod_x + 1
                y = (y + 1) % mod_y + 1
            else:
                x = (x + 1) % mod_x + 1
                y = (y + 1) % mod_y
        else:
            x += 1
        # 移动开始
        print(f"{x},{y}:{matrix_p[x][y].label}")#移动测试
        if matrix_p[x][y].label == empty_label:  # 如果移动到标签是空的情况下才会做判断
            count_p = 0.0
            for row in range(3):
                for col in range(3):
                    x_temp = x - 1
                    y_temp = y - 1
                    x_temp += col
                    y_temp += row
                    if (x_temp == tupel_x_y[1] + 1 and y_temp == tupel_x_y[0] + 1):  # 如果计算指针和最初的的x，y匹配则不计入这个
                        continue
                    elif matrix_p[x_temp][y_temp].label == ori_label:
                        count_p += 1.0
            happy_rate = count_p / 8.0
            # 计算完当前x,y的开心值了
            print(f"matrix_input[{x}][{y}] happy rate is :{happy_rate} -------------matrix_move_1point")
            if happy_rate > max_pro_happyrate:
                outOP_x_y = (y-1, x-1)  #？？？ unsure
                max_pro_happyrate = happy_rate
            if happy_rate >= happy_rate_k:
                unhappy_list, matrix_p = updata_8_point_happy_rate(matrix_p, x, y, tupel_x_y, happy_rate_k,
                                                                   unhappy_list)
                # visualization.plot_grid_label(matrix_p)  # -------------------------可视化测试
                matrix_p = matrix_p[1:-1, 1:-1]
                print(f"un happy list :\n {unhappy_list}")
                return unhappy_list, matrix_p
        max_k = max_k - 1
    if outOP_x_y != (tupel_x_y[0], tupel_x_y[1]):#不等于原本的
        unhappy_list, matrix_p = updata_8_point_happy_rate(matrix_p, outOP_x_y[1]+1, outOP_x_y[0]+1, tupel_x_y, happy_rate_k,
                                                       unhappy_list)
        matrix_p = matrix_p[1:-1, 1:-1]

        unhappy_list.append(outOP_x_y)
        print(f"un happy list :\n {unhappy_list}")
        # visualization.plot_grid_label(matrix_p)  # -------------------------可视化测试
        return unhappy_list, matrix_p
    else:
        unhappy_list.append(unhappy_list[0])
        unhappy_list.pop(0)
        matrix_p = matrix_p[1:-1, 1:-1]
        return unhappy_list, matrix_p



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
    unhappy_list = list()
    matrix_t = Matrix(label_r, N)
    # calculate_8_happy_rate(matrix_t.matrix, 1, 3) #测试单个点的快乐值
    matrix_res, unhappy_list = calculate_happy_rate(matrix_t, N, N, unhappy_list, 0.375)
    plot_iter = 5#多少次输出画面
    judge = 1
    iter_time = 0
    # visualization.plot_grid_label(matrix_res.matrix)
    # visualization.plot_grid_happy_rate(matrix_res.matrix, 0.375)
    while( judge ):
        print(f"--------------------------------------------------{iter_time}")
        unhappy_list, matrix_res.matrix = matrix_move_1point(matrix_res, unhappy_list[0], 0.375, unhappy_list)#看方法的注释，后面能添加单次迭代次数
        iter_time += 1
        if len(unhappy_list) == 0 or iter_time > 100:   #设置最大迭代次数
            break
        if iter_time % plot_iter == 0:
            unhappy_list = list()
            matrix_res, unhappy_list = calculate_happy_rate(matrix_t, N, N, unhappy_list, 0.375)
            visualization.plot_grid_label(matrix_res.matrix)
            visualization.plot_grid_happy_rate(matrix_res.matrix,0.375)
