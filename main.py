from scipy.io import loadmat
import pyvista as pv
import numpy as np
import cv2
# import verification
import scipy.io as scio
from ShowAxis_pv import ShowAxis_pv
from chessboard import plot_chessboard

#在左相机下生成特征点坐标，并对特征点进行画线
#显示plotter背景颜色
def plot_figure(plotter):
    plotter.background_color = (200, 200, 200)
    plotter.add_mesh(process_images(mtx_L, mtx_R, R_lr, t_lr, 100, 'Left_1.jpg', 'Right_1.jpg')[1], color='b', point_size=10, opacity=0.003, render_points_as_spheres=True,
                     label='ni')
    plotter.add_mesh(process_images(mtx_L, mtx_R, R_lr, t_lr, 100, 'Left_1.jpg', 'Right_1.jpg')[0], color='b', point_size=10, opacity=0.003, render_points_as_spheres=True,
                     label='ni')
    plotter.show_grid(color='black')
    # ShowAxis_pv(plotter, 70,R_lr,t_lr)
    cpos = plotter.show(cpos=[(2540.4629310301984, -528.5590045506597, -635.4590852621342),
 (-74.06932394962323, 177.03167228137278, 454.925619098772),
 (-0.21312927998695594, -0.9700332373797675, 0.11666802642870713)]
, return_cpos=True)
    print(cpos)
    # 保存图像
    # plotter.screenshot("output1.png")  # 图像将保存为 output.png

    # print(cpos)


def process_images(mtx_L, mtx_R, R_lr, t_lr, num, img_left_path, img_right_path):
    x_samples = np.linspace(-500, 500, num)
    y_samples = np.linspace(-500, 500, num)
    z_samples = np.linspace(500, 1000, int(num / 2))
    samples_l = np.array(np.meshgrid(x_samples, y_samples, z_samples)).T.reshape(-1, 3)
    samples_r = np.linalg.inv(R_lr).dot(samples_l.T - t_lr).T

    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    xiangsu_l = mtx_L.dot(samples_l.T)
    xiangsu_r = mtx_R.dot(R_lr.dot(samples_r.T)+t_lr)
    # print(R_lr.dot(samples_r.T)+t_lr)
    # sys.exit()
    print((xiangsu_l.T).shape)
    print((xiangsu_r.T).shape)
    # sys.exit()
    image_coords_normalized_L = xiangsu_l.T[:,:2]/ (xiangsu_l.T[:,2].reshape(-1,1))
    print(image_coords_normalized_L)
    # sys.exit()
    # image_coords_normalized_T_L = np.transpose(image_coords_normalized_L)

    image_coords_normalized_R = xiangsu_r.T[:,:2] / (xiangsu_r.T[:,2].reshape(-1,1))
    # image_coords_normalized_T_R = np.transpose(image_coords_normalized_R)

    maskL = (image_coords_normalized_L[:, 0] >= 0) & (image_coords_normalized_L[:, 0] < img_left.shape[1]) & \
            (image_coords_normalized_L[:, 1] >= 0) & (image_coords_normalized_L[:, 1] < img_left.shape[0])
    maskR = (image_coords_normalized_R[:, 0] >= 0) & (image_coords_normalized_R[:, 0] < img_right.shape[1]) & \
            (image_coords_normalized_R[:, 1] >= 0) & (image_coords_normalized_R[:, 1] < img_right.shape[0])

    valid_pointsL = samples_l[maskL]
    valid_pointsR = samples_r[maskR]

    left_trans_right = R_lr.dot(valid_pointsL.T) + t_lr
    xiangsu_common_rl = mtx_L.dot(valid_pointsR.T)
    xiangsu_common_lr = mtx_R.dot(left_trans_right)

    image_coords_normalized_common_rl = xiangsu_common_rl[:2] / xiangsu_common_rl[2]
    image_coords_normalized_comoon_lr = xiangsu_common_lr[:2] / xiangsu_common_lr[2]
    image_coords_normalized_common_rl_T = np.transpose(image_coords_normalized_common_rl)
    image_coords_normalized_comoon_lr_T = np.transpose(image_coords_normalized_comoon_lr)

    mask_common_lr = (image_coords_normalized_comoon_lr_T[:, 0] >= 0) & (
            image_coords_normalized_comoon_lr_T[:, 0] < img_right.shape[1]) & \
                     (image_coords_normalized_comoon_lr_T[:, 1] >= 0) & (
                                 image_coords_normalized_comoon_lr_T[:, 1] < img_right.shape[0])
    mask_common_rl = (image_coords_normalized_common_rl_T[:, 0] >= 0) & (
            image_coords_normalized_common_rl_T[:, 0] < img_left.shape[1]) & \
                     (image_coords_normalized_common_rl_T[:, 1] >= 0) & (
                                 image_coords_normalized_common_rl_T[:, 1] < img_left.shape[0])
    valid_common_lr = valid_pointsL[mask_common_lr]
    valid_common_rl = valid_pointsR[mask_common_rl]
    valid_common_l = valid_pointsL[~mask_common_lr]
    valid_common_r = valid_pointsR[~mask_common_rl]

    select_R = (valid_common_rl[:, -1] < 900) & (valid_common_rl[:, -1] > 400)
    valid_common_rl_select = valid_common_rl[select_R]

    valid_pointsL_panduan = R_lr.dot(valid_common_lr.T) + t_lr
    select_L = (valid_pointsL_panduan.T[:, -1] < 900) & (valid_pointsL_panduan.T[:, -1] > 400)
    valid_common_lr_select = valid_common_lr[select_L]

    valid_common_rl_out = valid_common_rl[~select_R]
    valid_common_lr_out = valid_common_lr[~select_L]

    return valid_common_lr_select, valid_common_rl_select
if __name__ == '__main__':

    # %% 载入双目结构参数
    data = scio.loadmat('Calib_Results_stereo.mat')  # 载入双目相机标定结果
    R_lr_vec = data['om']  # 左相机到右相机的旋转向量
    R_lr = cv2.Rodrigues(R_lr_vec)[0]  # 旋转向量转旋转矩阵
    t_lr = data['T']  # 左相机到右相机的平移矢量
    # %% 载入标定内外参数
    data_L = scio.loadmat('Calib_Results_left.mat')  # 载入左相机标定结果
    mtx_L = np.array([[data_L['fc'][0, 0], 0, data_L['cc'][0, 0]],  # 标定得到的左相机内参
                    [0, data_L['fc'][1, 0], data_L['cc'][1, 0]],
                    [0, 0, 1, ]])
    # %% 载入标定内外参数
    data_R = scio.loadmat('Calib_Results_right.mat')  # 载入右边相机标定结果
    mtx_R = np.array([[data_R['fc'][0, 0], 0, data_R['cc'][0, 0]],  # 标定得到的右相机内参
                    [0, data_R['fc'][1, 0], data_R['cc'][1, 0]],
                    [0, 0, 1, ]])

    # P=pv.Plotter(off_screen=True)
    P=pv.Plotter()
    P.resolution = (1000, 800)
    process_images(mtx_L, mtx_R, R_lr, t_lr, 100, 'Left_1.jpg', 'Right_1.jpg')
    ShowAxis_pv(P,140,R_lr,t_lr)
    plot_chessboard(P)
    plot_figure(P)

