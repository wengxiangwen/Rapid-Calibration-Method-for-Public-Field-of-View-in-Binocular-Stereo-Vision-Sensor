import pyvista as pv
import numpy as np
import cv2
def ShowAxis_pv(plotter, arrow_length,R_lr,t_lr):
    def pvshow_arrow(plotter, start, direction, color, opacity=0.5):
        arrow = pv.Arrow(start=start, direction=direction, scale='auto')  # scale为auto的情况下，箭头长度和direction相关
        plotter.add_mesh(arrow, color=color, opacity=opacity)


   # np.linalg.inv(R_lr).dot(samples_l.T - t_lr).T
    # 坐标系1：原点为（0，0，0）
    coordinate_points1 = np.array([[0, 0, 0],  # 原点
                                   [arrow_length, 0, 0],  # x
                                   [0, arrow_length, 0],  # y
                                   [0, 0, arrow_length], ])  # z
    pvshow_arrow(plotter, coordinate_points1[0], coordinate_points1[1], 'r', opacity=1)
    pvshow_arrow(plotter, coordinate_points1[0], coordinate_points1[2], 'g', opacity=1)
    pvshow_arrow(plotter, coordinate_points1[0], coordinate_points1[3], 'b', opacity=1)

    # print( coordinate_points2[0].reshape(3,1)-t_lr)
    coordinate_points2 = np.array([[0, 0, 0],  # 原点
                                   [arrow_length, 0, 0],  # x
                                   [0, arrow_length, 0],  # y
                                   [0, 0, arrow_length], ])  # z
    coordinate_points2 = (np.linalg.inv(R_lr).dot(coordinate_points2.T-t_lr)).T
    print(coordinate_points2)
    pvshow_arrow(plotter, coordinate_points2[0], coordinate_points2[1]- coordinate_points2[0] ,'r', opacity=1)
    pvshow_arrow(plotter, coordinate_points2[0], coordinate_points2[2]- coordinate_points2[0], 'g', opacity=1)
    pvshow_arrow(plotter, coordinate_points2[0], coordinate_points2[3]- coordinate_points2[0], 'b', opacity=1)


