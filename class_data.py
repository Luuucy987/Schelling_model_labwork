# -*- coding: utf-8 -*-
import numpy as np

N = 50  # 矩阵大小
empty_label = 0
error = 0
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


# x = np.arange(N)
# y = np.arange(N)
# X, Y = np.meshgrid(x, y)
# status = ["C1", "C2", 'white']  # 网格状态，生成三类格子，white为空白网格
# prob = [0.4, 0.4, 0.2]  # 三类格子的生成比例
#
#
# def init_z():
#     """初始化网格
#     :return: 返回创建的网格
#     """
#     Z = np.random.choice(a=status, size=(N ** 2), p=prob)
#     Z.shape = (N, N)
#     return Z
#
#
# def get_cell_happiness(Z, row, col):
#     """获取每个单元格的满意程度阈值
#     :Z: N*N np.array
#     :row: int行, col:int列
#     :return: happiness:int
#     """
#     if not Z.shape == (N, N):
#         Z.shape = (N, N)
#     if Z[row, col] == "white":
#         return np.NaN
#     same, count = 0, 0  # same为相同个数，count为有人的个数
#     left = 0 if col == 0 else col - 1
#     right = Z.shape[1] if col == Z.shape[1] - 1 else col + 2
#     top = 0 if row == 0 else row - 1
#     bottom = Z.shape[0] if row == Z.shape[0] - 1 else row + 2
#
#     for i in range(top, bottom):
#         for j in range(left, right):
#             if (i, j) == (row, col) or Z[i, j] == "white":
#                 continue
#             elif Z[i, j] == Z[row, col]:
#                 same += 1
#                 count += 1
#             else:
#                 count += 1
#     if not count == 0:
#         happiness = same / count
#     else:
#         happiness = 0
#     return happiness
#

# 注重代码规范
if __name__ == "__main__":
    test_point = point(1)
    print(f"test point lebal is {test_point.label}, happy_rate is {test_point.happy_rate}")
