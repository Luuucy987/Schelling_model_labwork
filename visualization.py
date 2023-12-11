# -*- coding: utf-8 -*-
import class_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_grid_label(grid:class_data.Matrix):
    """
    使用matplotlib生成基于label值的图像
    :param grid: class_data中matrix类
    :return: 生成的白、红、蓝的网格图像
    """
    # 将label参数单独生成一个二维矩阵
    labels_only = np.array([[point.label for point in row] for row in grid])


    # 定义颜色映射
    colors = ['white','red','blue']
    cmap = ListedColormap(colors)

    # 生成图像
    plt.imshow(labels_only, cmap=cmap, interpolation='none')
    plt.colorbar(ticks=[0, 1, 2])
    plt.show()

def plot_grid_happy_rate(grid:class_data.Matrix,threshold):
    """
    使用matplotlib生成基于happy_rate的图像
    :param grid: class_data中的Matrix类
    :param threshold: happt_rate的阈值
    :return: 生成的从白到黄再到灰的网格图
    """
    # 将happy_rate单独生成二维数组
    happy_rate_only = np.array([[point.happy_rate for point in row] for row in grid])


    # 定义颜色映射
    # colors = ['purple']
    # cmap = ListedColormap(colors)
    # 颜色的RGB编码
    color_list = [(1, 1, 1), (1, 1, 0), (0, 0, 0)]
    # 自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('white_to_yellow_to_black', color_list, N=256)
    # 生成图像
    plt.imshow(happy_rate_only.astype(float), cmap=cmap, interpolation='none',vmin=0,vmax=threshold)
    plt.colorbar()
    plt.show()

def save_grid_label_image(grid,iteration,save_folder='label_images'):
    """
    保存label生成的图像
    :param grid: class_data中的Matrix类
    :param iteration: 迭代次数
    :param save_folder: 保存的文件夹名称
    :return: 保存的文件夹和文件图片
    """
    # 创建保存文件夹（如果不存在）
    os.makedirs(save_folder,exist_ok=True)

    # 构建文件路径
    file_path = os.path.join(save_folder,f'iteration_{iteration}.png')

    labels_only = np.array([[point.label for point in row] for row in grid])
    # 定义颜色映射
    colors = ['white', 'red', 'blue']
    cmap = ListedColormap(colors)

    plt.imshow(labels_only, cmap=cmap, interpolation='none')
    plt.colorbar(ticks=[0, 1, 2])

    plt.title(f'Iteration{iteration}')
    # 保存图像到文件
    plt.savefig(file_path)
    plt.close()

def save_grid_happy_rate_image(grid,threshold,iteration,save_folder='happy_rate_images'):
    """
    保存基于happy_rate生成的图像
    :param grid:class_data中的Matrix类
    :param threshold: happy_rate的阈值
    :param iteration: 迭代次数
    :param save_folder: 保存图片的文件夹名称
    :return: 保存的文件夹和文件图片
    """
    # 创建保存文件夹（如果不存在）
    os.makedirs(save_folder,exist_ok=True)

    # 构建文件路径
    file_path = os.path.join(save_folder,f'iteration_{iteration}.png')

    happy_rate_only = np.array([[point.happy_rate for point in row] for row in grid])
    color_list = [(1, 1, 1), (1, 1, 0), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('white_to_yellow_to_black', color_list, N=256)
    plt.imshow(happy_rate_only.astype(float), cmap=cmap, interpolation='none', vmin=0, vmax=threshold)
    plt.colorbar()

    plt.title(f'Iteration{iteration}')
    # 保存图像到文件
    plt.savefig(file_path)
    plt.close()

if __name__ == "__main__":
    # 参数设置
    label_list = [0, 1, 2]  # 标签列表
    label_weight = [0.5, 0.2, 0.3]  # n个标签的权值列表【int】类型列表，第0个表示 标签为空的权值
    label_n = 30  # Maxtir size n * n
    label_rand = class_data.random_label(label_n, label_n, _label_list=label_list, _weight=label_weight)
    maxtir_t = class_data.Matrix(label_rand, label_n)
    test_list = list()
    matrix_res, test_list = class_data.calculate_happy_rate(maxtir_t,label_n,label_n,test_list,0.375)
    print(maxtir_t.matrix[0][0].label)

    # 可视化初始状态
    plot_grid_label(maxtir_t.matrix)
    plot_grid_happy_rate(matrix_res.matrix,0.375)
    save_grid_label_image(maxtir_t.matrix,1)
    save_grid_happy_rate_image(matrix_res.matrix,0.375,1)