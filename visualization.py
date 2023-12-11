# -*- coding: utf-8 -*-
import class_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import os


# def test_fuction(test: class_data.Matrix):
#     return 0





# def initialize_grid(size:int, proportions:float):
#     """
#     使用每个标签的给定比例初始化代理网格
#     :param size:整型，表示网格的大小
#     :param proportions:浮点型，表示初始时每个类型的比例，是一个列表
#     """
#     # 初始化网络
#     grid = np.random.choice([0, 1, 2], size=(size, size), p=proportions)
#     return grid

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

# def schelling_model(grid, similarity_threshold, max_iterations=100):
#     """
#     Run the Schelling model simulation.
#     """
#     size = grid.shape[0]
# 
#     for iteration in range(max_iterations):
#         for i in range(size):
#             for j in range(size):
#                 if grid[i, j] == 0:
#                     continue  # Skip empty cells
# 
#                 neighbors = get_neighbors(grid, i, j)
#                 similar_neighbors = sum(neighbors == grid[i, j])
# 
#                 if similar_neighbors / len(neighbors) < similarity_threshold:
#                     # Move the agent to a random empty cell
#                     empty_cells = np.argwhere(grid == 0)
#                     if len(empty_cells) > 0:
#                         new_location = empty_cells[np.random.choice(len(empty_cells))]
#                         grid[new_location[0], new_location[1]] = grid[i, j]
#                         grid[i, j] = 0
# 
#     return grid

# def get_neighbors(grid, i, j):
#     """
#     Get the neighbors of a cell in the grid.
#     """
#     size = grid.shape[0]
#     neighbors = []
# 
#     for x in range(max(0, i-1), min(size, i+2)):
#         for y in range(max(0, j-1), min(size, j+2)):
#             if x != i or y != j:
#                 neighbors.append(grid[x, y])
# 
#     return np.array(neighbors)
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
    # grid_size = 5
    # proportions = [0.5, 0.3, 0.2]  # 三个标签的初始比例
    # similarity_threshold = 0.375
    #
    # # 初始化网格
    # grid = initialize_grid(grid_size, proportions)

    # 可视化初始状态
    plot_grid_label(maxtir_t.matrix)
    plot_grid_happy_rate(matrix_res.matrix,0.375)
    save_grid_label_image(maxtir_t.matrix,1)
    save_grid_happy_rate_image(matrix_res.matrix,0.375,1)
    # 运行谢林模型
    # final_grid = schelling_model(grid, similarity_threshold)

    # 可视化最终状态
    # plot_grid(final_grid)
