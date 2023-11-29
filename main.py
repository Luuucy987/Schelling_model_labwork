# -*- coding: utf-8 -*-

import numpy as np


# 定义自定义类
class CustomClass:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"CustomClass({self.value})"


# 设置 n 的大小
n = 3

# 创建一个包含对象实例的二维数组
custom_objects_array = np.array([
    [CustomClass(i * n + j) for j in range(n)]  # 内层列表生成式 先生成n次，相当于先实现 CustomClass[n]
    for i in range(n)  # 外层列表生成式    实现CustomClass[n][n]
])
# 创建一个包含对象实例的一维数组
custom_objects_array_one = np.array(
    [CustomClass(n + j) for j in range(n)]  # 内层列表生成式 实现 CustomClass[n]
)

# 打印结果
print(custom_objects_array)
print("****************")
print(custom_objects_array_one)
obj: custom_objects_array = custom_objects_array[1, 2]
print(type(obj))
