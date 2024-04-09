import sys

from scipy.io import loadmat
import pyvista as pv
import numpy as np
import cv2
import matplotlib.pyplot as plt

#在左相机下生成特征点坐标，并对特征点进行画线
def plot_chessboard(plotter):
    data = loadmat('Calib_Results_left.mat')
    Rc_1 = data['Rc_1']
    Rc_2 = data['Rc_2']
    Rc_3 = data['Rc_3']
    Rc_4 = data['Rc_4']
    Tc_1 = data['Tc_1']
    Tc_2 = data['Tc_2']
    Tc_3 = data['Tc_3']
    Tc_4 = data['Tc_4']
    X_t = data['X_1']
    points_Camerica1 = Rc_1.dot(X_t) + Tc_1
    points_Camerica2 = Rc_2.dot(X_t) + Tc_2
    points_Camerica3 = Rc_3.dot(X_t) + Tc_3
    points_Camerica4 = Rc_4.dot(X_t) + Tc_4
    indices = [0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40,47, 48, 55, 56, 63, 64,71,72,79,80,87]


    # print(points)
    # sys.exit()
    # 遍历每一行和列，计算每个特征点的坐标并添加到列表中
    for i in range(len(indices) - 1):
     # line_start1 = points_Camerica1.T[indices[i]]
     # line_start2 = points_Camerica2.T[indices[i]]
     # line_start3 = points_Camerica3.T[indices[i]]
     # line_start4 = points_Camerica4.T[indices[i]]
     # line_end1 = points_Camerica1.T[indices[i + 1]]
     # line_end2 = points_Camerica2.T[indices[i + 1]]
     # line_end3 = points_Camerica3.T[indices[i + 1]]
     # line_end4 = points_Camerica4.T[indices[i + 1]]
     # plotter.add_lines(np.array([line_start1, line_end1]), color='g', width=5, label=None)
     # plotter.add_lines(np.array([line_start2,  line_end2]), color='g', width=5, label=None)
     # plotter.add_lines(np.array([line_start3, line_end3]), color='g', width=5, label=None)
     # plotter.add_lines(np.array([line_start4, line_end4]), color='g', width=5, label=None)

     plotter.add_mesh(points_Camerica1.T, color=[154, 205, 50], point_size=10, opacity=1, render_points_as_spheres=True)
     plotter.add_mesh(points_Camerica2.T, color=[245, 234, 139], point_size=10, opacity=1, render_points_as_spheres=True)
     plotter.add_mesh(points_Camerica3.T, color=[255, 215, 0], point_size=10, opacity=1, render_points_as_spheres=True)
     plotter.add_mesh(points_Camerica4.T, color=[255, 128, 128], point_size=10, opacity=1, render_points_as_spheres=True)
     # print(points_Camerica1.T)

