# import the useful packages

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import open3d as o3d
import copy
from tqdm import tqdm
import time
import pandas as pd
import math

# because it is a bit complicated to install the python-pcl, define a ICP function by ourselves
from icp import icp


# define a data loader to load the img, depth and camera pose from the tartan dataset
def tartan_data_loader(path: str, position: str, start_num: int = 0, end_num: int = 5):
    """
    Load the tartanair data set

    Args:
        _path (str): the path of the dataset
        _position (str): left or right
        _start_num (int): the start idx of the sequence
        _end_num (int): the end idx of the sequence, if it is over1000 it will be set as the length of the seauence

    Returns:
        _img (np.ndarray): the color img <n, 480, 640, 3>
        _depth (np.ndarray): the depth img <n, 480, 640, 3>
        _camera_pose (np.ndarray): the camera pose <n, 7>
    """
    # Set the path
    _image_dir = path + "/image_" + position
    _depth_dir = path + "/depth_" + position
    _pose_dir = path + "/pose_" + position + ".txt"

    img_sequence = np.zeros((0, 480, 640, 3))  # Create a empty array to store the img

    _image_dir_sequences = os.listdir(_image_dir)
    end_num = len(_image_dir_sequences) if end_num > 1000 else end_num

    _image_dir_sequences.sort(key=lambda x: int(x.split("_")[0]))  # Solve the problem that the list is not in order

    # _test_video = cv2.VideoWriter("VideoTest.avi", cv2.VideoWriter_fourcc("I", "4", "2", "0"), 5, (640, 480))

    print("Load the color image:")
    for _img_name in tqdm(_image_dir_sequences[start_num:end_num]):
        _img = cv2.imread(_image_dir + "/" + _img_name)
        # _test_video.write(_img)
        _img = np.expand_dims(_img, axis=0)  # expend the dim for the concatenate
        img_sequence = np.concatenate((img_sequence, _img), axis=0)

    # _test_video.release()
    # cv2.destroyAllWindows()

    depth_sequence = np.zeros((0, 480, 640))  # Create a empty array to store the depth

    _depth_dir_sequences = os.listdir(_depth_dir)
    _depth_dir_sequences.sort(key=lambda x: int(x.split("_")[0]))  # Solve the problem that the list is not in order

    print("Load the depth image:")
    for _depth_name in tqdm(_depth_dir_sequences[start_num:end_num]):
        _depth = np.load(_depth_dir + "/" + _depth_name)
        _depth = np.expand_dims(_depth, axis=0)  # expend the dim for the concatenate
        depth_sequence = np.concatenate((depth_sequence, _depth), axis=0)

    pose_sequence = np.loadtxt(_pose_dir)[start_num:end_num]

    return img_sequence, depth_sequence, pose_sequence


# define a small function to draw the depth and color ilg together by using the plt
def draw_depth_color(depthimg: np.ndarray, colorimg: np.ndarray):
    """
    plot the depth and color image together

    Args:
      _depthimg (np.ndarray): the depth image
      _colorimg (np.ndarray): the color image

    Returns:
      plot

    """
    print("Check the depth and color image", "\n")
    plt.figure(figsize=(15, 20))

    plt.subplot(1, 2, 1)
    plt.imshow(depthimg, cmap=plt.cm.gray)  # set to the gray plot
    plt.title("Depth Image")

    plt.subplot(1, 2, 2)
    _imgRGB = cv2.cvtColor(colorimg.astype("uint8"), cv2.COLOR_BGR2RGB)  # change the color channel
    plt.imshow(_imgRGB.astype(np.float64) / 255.0)
    plt.title("Color Image")

    plt.show()

    return 0


# define a function to homogenize the coordinate
def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate

    Args:
        points (np.ndarray): 2D or 3D coordinate <float: num_points, num_dim>

    Returns:
        np.ndarray: 3D or 4D homogeneous coordinate <float: num_points, num_dim + 1>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# define a function to multiply two quaternions
def quaternion_multiply(quaternion0: np.ndarray, quaternion1: np.ndarray) -> np.ndarray:
    """
     Multiply two quaternions

    Args:
        quaternion0 (np.ndarray):  <4, >
        quaternion1 (np.ndarray):  <4, >

    Returns:
        np.ndarray:  <1, 4>
    """
    _x0, _y0, _z0, _w0 = quaternion0
    _x1, _y1, _z1, _w1 = quaternion1

    ans = np.array(
        [
            _w0 * _x1 + _x0 * _w1 + _y0 * _z1 - _z0 * _y1,
            _w0 * _y1 - _x0 * _z1 + _y0 * _w1 + _z0 * _x1,
            _w0 * _z1 + _x0 * _y1 - _y0 * _x1 + _z0 * _w1,
            _w0 * _w1 - _x0 * _x1 - _y0 * _y1 - _z0 * _z1,
        ],
        dtype=np.float64,
    ).T

    return ans


# define a function to inverse a quaternion
def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    """
       Inverse a quaternion

    Args:
        quaternion (np.ndarray):  <4, >

    Returns:
        the inverse of the quaternion  <4, >
    """
    x0, y0, z0, w0 = quaternion
    return np.array([-x0, -y0, -z0, w0]) / (w0 * w0 + x0 * x0 + y0 * y0 + z0 * z0)


# define a function to get the pointcloud from the depth image
def pointcloud_from_depth(
    depthimg: np.ndarray, camera_paras: np.ndarray, flatten: bool = True, threshold: int = 10000
) -> np.ndarray:
    """Create point clouds from the depth image

    Args:
        _depthimg (np.ndarray): the depth image
        _camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)

    Returns:
        _pointclouds (np.ndarray): pointclouds <float: num_points, 3>
    """
    _fx, _fy, _cx, _cy = camera_paras

    # print(np.max(_depthimg))
    _depthimg = np.where(depthimg > threshold, threshold, depthimg)
    # print(np.max(_depthimg))

    _h, _w = np.mgrid[0 : _depthimg.shape[0], 0 : _depthimg.shape[1]]
    _z = _depthimg
    _x = (_w - _cx) * _z / _fx
    _y = (_h - _cy) * _z / _fy

    pointclouds = np.dstack((_x, _y, _z)) if flatten == False else np.dstack((_x, _y, _z)).reshape(-1, 3)

    return pointclouds


# define a function to get the transformation matrix from the pose
def get_T_from_pose(camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1])) -> np.ndarray:
    """
    Get the transformation matrix wTc from the pose

    Args:
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]).

    Returns:
        _wTc (np.ndarray): transformation matrix wTc <4, 4>
    """
    # Cause we already transform the origin frame from NED frame to its own
    # No need to transform here
    _t, _R_quaternion = camera_pose[:3], camera_pose[3:]
    _wRc_mat = R.from_quat(_R_quaternion).as_matrix()
    # Or we could use the o3d.geometry.get_rotation_matrix_from_quaternion()
    # The camera motion is defined in the NED frame

    # _NEDRc_mat = R.from_euler('yzx', [90, -90, 0], degrees=True).as_matrix()

    # _wRc_mat = _wRNED_mat.dot(_NEDRc_mat)

    wTc = np.eye(4)
    wTc[:3, :3] = _wRc_mat
    wTc[:3, 3] = _t.T

    return wTc


# define a function to transfer the pointcloud from the camera frame to the world frame
def camera2world(_pointclouds: np.ndarray, _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1])) -> np.ndarray:
    """
    Change the pointclouds from camera frame to the world frame

    Args:
        _pointclouds (np.ndarray):
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]). The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.

    Returns:
        _pointclouds (np.ndarray):
    """
    _wTc = get_T_from_pose(_camera_pose)
    _pointclouds = _wTc.dot(homogenize(_pointclouds).T).T

    return _pointclouds[:, :3]


# define a function to get the inverse of a transformation matrix
def inverse_transformation(T: np.ndarray) -> np.ndarray:
    """
     Get the inverse of the transformation

    Args:
        _T (np.ndarray): <4, 4>

    Returns:
        _T_inv np.ndarray: <4, 4>
    """
    _R = T[:3, :3]
    _t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = _R.T
    T_inv[:3, 3] = -(_R.T).dot(_t)

    return T_inv


# define a function to move the pose sequence to a given frame,
# we assume the input is a sequence of the pose, maybe it works not good for the iterator
def move_pose(pose_sequence: np.ndarray, given_frame: np.ndarray):
    """
    Move the pose sequence to a given frame
    ! the process of the quaternion
    Args:
        _pose_sequence (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form
        _given_frame (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form

    Returns:
        _pose_sequence_new (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form
    """
    _pose_sequence_new = copy.deepcopy(pose_sequence)

    # move the orientation
    for i in range(len(pose_sequence)):
        _pose_sequence_new[i, 3:] = quaternion_multiply(pose_sequence[i, 3:], quaternion_inverse(given_frame[3:]))

    _wTc0 = np.eye(4)
    _wTc0[:3, :3] = R.from_quat(given_frame[3:]).as_matrix()
    _wTc0[:3, 3] = given_frame[:3].T

    # move the position
    _c0Tw = inverse_transformation(_wTc0)  # Using the np.linalg.inv does not work well

    _t_sequence = _c0Tw.dot(homogenize(_pose_sequence_new[:, :3]).T).T

    return np.hstack((_t_sequence[:, :3], _pose_sequence_new[:, 3:]))


# define a function to move the rotation along x and z axis of the camera, cause we condier the motion is a 2D motion,
# we assume the input is a sequence of the pose, maybe it works not good for the iterator
def remove_rotation_xz(pose_sequence: np.ndarray) -> np.ndarray:
    """
     Move the useless rotation from the pose

    Args:
        _pose_sequence (np.ndarray): <nums, 7> we assume the input is the whole pose

    Returns:
        _pose_sequence_new (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form
    """
    pose_sequence_new = copy.deepcopy(pose_sequence)
    for i in range(len(pose_sequence)):
        rx, ry, rz = R.from_quat(pose_sequence[i, 3:]).as_euler("xyz", degrees=False)
        pose_sequence_new[i, 3:] = R.from_euler("xyz", [0, ry, 0], degrees=False).as_quat()

    return pose_sequence_new


# define a function to visualize the pointcloud without the color
def draw_pointclouds_inblack(
    _pointclouds: np.ndarray,
    _window_name: str = "Pointcloud",
    _background_color: np.ndarray = np.asarray([0, 0, 0]),
    _pointcloud_color: np.ndarray = np.asarray([1, 1, 1]),
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 0]),
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """
    To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to 'Pointcloud'.
        _background_color (np.ndarray, optional): Defaults to np.asarray([0, 0, 0]).
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 0]). The camera pose in the world frame <1, 7> rotaiion vector in quaternion form.
        _pointcloud_color (np.ndarray, optional): Defaults to np.asarray([1, 1, 1]).
        _frame_size (int, optional):  Defaults to 1.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """

    print("Visualize the pointcloud in white:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _vis.get_render_option().background_color = _background_color  # set the color of the background

    _pointclouds_o3d = o3d.geometry.PointCloud()
    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_o3d.paint_uniform_color(_pointcloud_color)  # set the color of the pointclouds
    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()
    _vis_controller.set_front((0, 0, -1))
    _vis_controller.set_lookat((1, 0, 0))
    _vis_controller.set_up((0, -1, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# define a function to visualize the pointcloud with the color
def draw_pointclouds_incolor(
    _pointclouds: np.ndarray,
    _pointclouds_color: np.ndarray,
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1]),
    _window_name: str = "Pointcloud",
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to'Pointcloud'
        _frame_size (int, optional):  Defaults to 1.
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 0]). The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """
    print("Visualize the pointcloud in color:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _pointclouds_o3d = o3d.geometry.PointCloud()

    # _vis.get_render_option().background_color = [0, 0, 0] # set the backgroud to black
    # Set a threshold to filter some large depth pointclouds
    # ? Does not work, idk why but maybe modify the depth image directly is better
    # _threshold = 0.5
    # print(np.max(_pointclouds[:, 1]))
    # _pointclouds[:, 1] = np.where(_pointclouds[:, 1] < _threshold, _pointclouds[:, 1], _threshold)
    # print(np.max(_pointclouds[:, 1]))

    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_color_RGB = cv2.cvtColor(_pointclouds_color.astype("uint8"), cv2.COLOR_BGR2RGB)
    _pointclouds_color_RGB = (
        _pointclouds_color_RGB.reshape(-1, 3).astype(np.float64) / 255.0
    )  # the o3d is different as plt, the color should go to [0, 1]
    _pointclouds_o3d.colors = o3d.utility.Vector3dVector(_pointclouds_color_RGB)

    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()  # choose a BEV angle
    _vis_controller.set_front((0, -1, 0))
    _vis_controller.set_lookat((0, 0, 1))
    _vis_controller.set_up((-1, 0, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# define a function to correct the pointcloud(remove the rotation of x and z) based on the camera pose
def correct_pc_rotationxz(pointcloud: np.ndarray, camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1])):
    """
    Correct the pointcloud(remove the rotation of x and z) based on the camera pose

    Args:
        _pointcloud (np.ndarray): <nums, 3>
        _camera_pose (np.ndarray, optional): The camera pose in the world frame <nums, 7> rotaiion vector in
        quaternion form. Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]).
                                            Note: it should be the camera pose relative to the the pose 0

    Returns:
        _pointcloud: <nums, 3>
    """
    _r = R.from_quat(camera_pose[3:]).as_euler("xyz")
    _r[1] = 0  # keep the rotation along y
    _R_mat = R.from_euler("xyz", _r).as_matrix()

    pointcloud = _R_mat.T.dot(pointcloud.T).T

    return pointcloud


# define a function to create the BEV image from the pointcloud with color
def get_BEV_from_pointcloud(pointcloud: np.ndarray, BEV_size: np.ndarray, pc_shape: np.ndarray):
    """
      Create the BEV image from the pointcloud with color
    Args:
        _pointcloud (np.ndarray): <nums, 6>
        _size (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    start = time.time()

    _BEV_image = np.repeat(255 * np.ones(BEV_size)[:, :, None], 3, axis=-1)  # Maybe the background in white is better

    _Xindex = pointcloud[:, 0] // pc_shape[0]  # Becasue here, some information are lost
    _Zindex = pointcloud[:, 2] // pc_shape[1]

    _df = pd.DataFrame({"Xindex": _Xindex, "Y": pointcloud[:, 1], "Zindex": _Zindex})
    _df = (
        _df.groupby(["Xindex", "Zindex"])
        .idxmin()
        .reset_index()  # test the performance with the min method, min should be correct
        # _df.groupby(["Xindex", "Zindex"]).idxmax().reset_index()
    )  # use idxmax() or idxmin() to change to the opposite view

    _index = np.array(_df).astype(np.int32)

    _BEV_image[_index[:, 0], _index[:, 1]] = pointcloud[_index[:, 2], 3:]

    BEV_image_RGB = cv2.cvtColor(_BEV_image.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("In", time.time() - start, "seconds get the BEV")

    # plt.imshow(_BEV_image_RGB.astype(np.float64) / 255.0)
    # plt.show()

    return BEV_image_RGB


# define a function to create the BEV from the depth and color image
def get_BEV_from_depthandcolor(
    depth: np.ndarray,
    img: np.ndarray,
    camera_pose: np.ndarray,
    camera_paras: np.ndarray,
    BEV_size: np.ndarray = np.asarray((240, 320)),
    threshold: int = 10000,
    n_slice: int = 3,
    pc_range: np.ndarray = np.asarray((15, 20)),
    pc_shift: float = 10.0,
):
    """
    Create the BEV from the pointcloud with the color
    Args:
        depth (np.ndarray): <480, 640, 1>
        img (np.ndarray): <480, 640, 3>
        camera_pose (np.ndarray): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
                                           Note: it should be the camera pose relative to the the pose 0
        camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)
        size (tuple): <w, h>
        threshold (int):
        n_slice (int):

    Returns:
        BEV_image: <n_slice, w, h, 3>
    """

    _pointcloud = pointcloud_from_depth(
        depth, camera_paras, threshold=threshold
    )  # The frame is based on the camera frame! So we actually take the x and z, not z. We should know we are in which frmae everytime.
    _pointcloud = correct_pc_rotationxz(_pointcloud, camera_pose)

    # print(_pointcloud)

    _pointcloud = np.hstack((_pointcloud, img.reshape(-1, 3)))  # <nums, 6>

    # _min = np.min(_pointcloud[:, [0, 2]], axis=0) - 1e-8
    # _max = np.max(_pointcloud[:, [0, 2]], axis=0) + 1e-8

    # print('min is', _min, 'max is', _max)
    _pointcloud[:, 0] = _pointcloud[:, 0] + pc_shift  # make sure all points are positive for the BEV img

    # _pointcloud = _pointcloud[((0 < _pointcloud[:, 0] < _range[0]) & (0 < _pointcloud[:, 2] < _range[1]))]
    _pointcloud = _pointcloud[
        ((_pointcloud[:, 0] < pc_range[0]) & (_pointcloud[:, 0] > 0) & (_pointcloud[:, 2] < pc_range[1]))
    ]

    print("Pointcloud shape is ", _pointcloud.shape)

    _shape = pc_range / BEV_size
    # print("shape is", _shape)

    # print('shape is', _shape)
    # print(np.max(_pointcloud[:, 1])) # It seems that we do not need the sort() here, the pointcloud is already
    # sorted, but for the extra safe, i did the sort again
    _pointcloud = _pointcloud[np.argsort(_pointcloud[:, 1])]

    # Check if the Pointcloud could be split equaly
    _pointcloud_list = np.array_split(_pointcloud, n_slice, axis=0)

    BEV_image_RGB = np.zeros((n_slice, BEV_size[0], BEV_size[1], 3))

    for i in tqdm(range(n_slice)):
        # print(_pointcloud_list[i].shape)
        BEV_image_RGB[i] = get_BEV_from_pointcloud(_pointcloud_list[i], BEV_size, _shape)

    return BEV_image_RGB


# define a function to change the RGBBEV to a 2D pointcloud
def BEV_to_2dpc(BEV_image: np.ndarray, pc_shape: np.ndarray, pc_shift: float):
    """
    define a function to change the RGBBEV to a 2D pointcloud

    Args:
        _BEV_image (np.ndarray): _description_

    Returns:
        pc_2d: _description_
    """
    pc_2d = np.where(BEV_image < 254.9999, 255.0, 0.0)
    pc_2d = pc_2d[:, :, 0]

    # print("SHAPE is ", _shape)

    pc_2d = np.asarray(np.where(pc_2d[:, :] == 255.0)).astype(np.float64).T

    pc_2d[:, 0] = pc_2d[:, 0] * pc_shape[0]
    pc_2d[:, 1] = pc_2d[:, 1] * pc_shape[1]
    pc_2d[:, 0] = pc_2d[:, 0] - pc_shift

    return pc_2d


# define a funcion to get two 2d pointclouds from two 2d BEV, matched by using ORB feature
def ORB_BEV_to_2dpc(BEV_0: np.ndarray, BEV_1: np.ndarray, pc_shape: np.ndarray, pc_shift: float):

    # Extract the orb
    _orb = cv2.ORB_create()
    print(BEV_0.shape)
    _kp0, _des0 = _orb.detectAndCompute(BEV_0.astype(np.uint8), None)
    _kp1, _des1 = _orb.detectAndCompute(BEV_1.astype(np.uint8), None)

    # match kpts
    _bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    _matches = _bf.match(_des0, _des1)

    _min_distance = _matches[0].distance

    for _x in _matches:
        if _x.distance < _min_distance:
            _min_distance = _x.distance

    _good_match = []
    for _x in _matches:
        if _x.distance <= max(2 * _min_distance, 30):
            _good_match.append(_x)

    # organize key points into matrix, each row is a point
    # The return of the .pt is the x and y of the feature
    # pc_2D_0 = np.array([_kp0[m.queryIdx].pt for m in _good_match]).reshape((-1, 2))  # shape: <num_pts, 2>
    # pc_2D_1 = np.array([_kp1[m.trainIdx].pt for m in _good_match]).reshape((-1, 2))  # shape: <num_pts, 2>

    pc_2D_0 = np.array([_kp0[m.queryIdx].pt for m in _matches]).reshape((-1, 2))  # shape: <num_pts, 2>
    pc_2D_1 = np.array([_kp1[m.trainIdx].pt for m in _matches]).reshape((-1, 2))  # shape: <num_pts, 2>

    pc_2D_0[:, 0] = pc_2D_0[:, 0] * pc_shape[0]
    pc_2D_0[:, 1] = pc_2D_0[:, 1] * pc_shape[1]
    pc_2D_0[:, 0] = pc_2D_0[:, 0] - pc_shift

    pc_2D_1[:, 0] = pc_2D_1[:, 0] * pc_shape[0]
    pc_2D_1[:, 1] = pc_2D_1[:, 1] * pc_shape[1]
    pc_2D_1[:, 0] = pc_2D_1[:, 0] - pc_shift

    # res = cv2.drawMatches(image_0[i].astype(np.uint8), _kp0, image_1[i].astype(np.uint8), _kp1, matches, outImg=None)
    # cv2.namedWindow("Match Result", 0)
    # cv2.resizeWindow("Match Result", 1000, 1000)
    # cv2.imshow("Match Result", res)
    # cv2.waitKey(0)

    return pc_2D_0, pc_2D_1


# a test function
def test_draw_pointclouds(
    _pointclouds: np.ndarray,
    _pointclouds_ry: np.ndarray,
    _pointclouds_color: np.ndarray,
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1]),
    _window_name: str = "Pointcloud",
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to'Pointcloud'
        _frame_size (int, optional):  Defaults to 1.
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]). The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """
    print("Visualize the pointcloud in color:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _pointclouds_o3d = o3d.geometry.PointCloud()

    # _vis.get_render_option().background_color = [0, 0, 0] # set the backgroud to black

    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_color_RGB = cv2.cvtColor(_pointclouds_color.astype("uint8"), cv2.COLOR_BGR2RGB)
    _pointclouds_color_RGB = (
        _pointclouds_color_RGB.reshape(-1, 3).astype(np.float64) / 255.0
    )  # the o3d is different as plt, the color should go to [0, 1]
    _pointclouds_o3d.colors = o3d.utility.Vector3dVector(_pointclouds_color_RGB)

    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    # For the test
    _pointclouds_o3d_ry = o3d.geometry.PointCloud()

    # _vis.get_render_option().background_color = [0, 0, 0] # set the backgroud to black
    _pointclouds_o3d_ry.points = o3d.utility.Vector3dVector(_pointclouds_ry)

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_pointclouds_o3d_ry)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()
    _vis_controller.set_front((0, 0, -1))
    _vis_controller.set_lookat((1, 0, 0))
    _vis_controller.set_up((0, -1, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# a test function
def get_frontview_from_depthandcolor(
    _depth: np.ndarray,
    _img: np.ndarray,
    _camera_pose: np.ndarray,
    _camera_paras: np.ndarray,
    _size: np.ndarray = (240, 320),
    _threshold: int = 10000,
    _N_slice: int = 3,
):
    """
    Create the BEV from the pointcloud with the color
    Args:
        _depth (np.ndarray): <480, 640, 1>
        _img (np.ndarray): <480, 640, 3>
        _camera_pose (np.ndarray): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
                                            Note: it should be the camera pose relative to the the pose 0
        _camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)
        _size (tuple): _description_
        _threshold (int):
        _N_slice (int):

    Returns:
        _BEV_image: <size>
    """
    start = time.time()

    _pointcloud = pointcloud_from_depth(
        _depth, _camera_paras, _threshold=_threshold
    )  # The frame is based on the camera frame! So we actually take the x and z, not z. We should know we are in which frmae everytime.
    _pointcloud = correct_pc_rotationxz(_pointcloud, _camera_pose)

    _pointcloud = np.hstack((_pointcloud, _img.reshape(-1, 3)))  # <nums, 6>

    _min = np.min(_pointcloud[:, [1, 0]], axis=0) - 1e-8
    _max = np.max(_pointcloud[:, [1, 0]], axis=0) + 1e-8

    print("min is", _min, "max is", _max)

    _shape = (_max - _min) / _size

    print("shape is", _shape)

    _BEV_image = np.repeat(255 * np.ones(_size)[:, :, None], 3, axis=-1)  # Maybe the background in white is better

    _pointcloud[:, [1, 0]] = _pointcloud[:, [1, 0]] - _min
    _Xindex = _pointcloud[:, 0] // _shape[1]
    _Yindex = _pointcloud[:, 1] // _shape[0]

    _df = pd.DataFrame({"Xindex": _Xindex, "Yindex": _Yindex, "Zindex": _pointcloud[:, 2]})  # ?
    print(_df)
    _df = _df.groupby(["Yindex", "Xindex"]).idxmin().reset_index()  # ?
    print(_df)

    _index = np.array(_df).astype(np.int32)

    print(_index.shape)

    print(_index[:, 2])

    _BEV_image[_index[:, 0], _index[:, 1]] = _pointcloud[_index[:, 2], 3:]  # ?
    _BEV_image_RGB = cv2.cvtColor(_BEV_image.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("In", time.time() - start, "seconds get the BEV")

    plt.imshow(_BEV_image_RGB.astype(np.float64) / 255.0)
    plt.show()

    return 0


# a test function
def get_sideview_from_depthandcolor(
    _depth: np.ndarray,
    _img: np.ndarray,
    _camera_pose: np.ndarray,
    _camera_paras: np.ndarray,
    _size: np.ndarray = (240, 320),
    _threshold: int = 10000,
    _N_slice: int = 3,
):
    """
    Create the BEV from the pointcloud with the color
    Args:
        _depth (np.ndarray): <480, 640, 1>
        _img (np.ndarray): <480, 640, 3>
        _camera_pose (np.ndarray): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
                                            Note: it should be the camera pose relative to the the pose 0
        _camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)
        _size (tuple): _description_
        _threshold (int):
        _N_slice (int):

    Returns:
        _BEV_image: <size>
    """
    start = time.time()

    _pointcloud = pointcloud_from_depth(
        _depth, _camera_paras, _threshold=_threshold
    )  # The frame is based on the camera frame! So we actually take the x and z, not z. We should know we are in which frmae everytime.
    _pointcloud = correct_pc_rotationxz(_pointcloud, _camera_pose)

    _pointcloud = np.hstack((_pointcloud, _img.reshape(-1, 3)))  # <nums, 6>

    _min = np.min(_pointcloud[:, [1, 2]], axis=0) - 1e-8
    _max = np.max(_pointcloud[:, [1, 2]], axis=0) + 1e-8

    print("min is", _min, "max is", _max)

    _shape = (_max - _min) / _size

    print("shape is", _shape)

    _BEV_image = np.repeat(255 * np.ones(_size)[:, :, None], 3, axis=-1)  # Maybe the background in white is better

    _pointcloud[:, [1, 2]] = _pointcloud[:, [1, 2]] - _min
    _Yindex = _pointcloud[:, 1] // _shape[0]
    _Zindex = _pointcloud[:, 2] // _shape[1]

    _df = pd.DataFrame({"Xindex": _pointcloud[:, 0], "Yindex": _Yindex, "Zindex": _Zindex})  # ?
    print(_df)
    _df = _df.groupby(["Yindex", "Zindex"]).idxmin().reset_index()  # ?
    print(_df)

    _index = np.array(_df).astype(np.int32)

    print(_index.shape)

    print(_index[:, 2])

    _BEV_image[_index[:, 0], _index[:, 1]] = _pointcloud[_index[:, 2], 3:]  # ?
    _BEV_image_RGB = cv2.cvtColor(_BEV_image.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("In", time.time() - start, "seconds get the BEV")

    plt.imshow(_BEV_image_RGB.astype(np.float64) / 255.0)
    plt.show()

    return 0


# a test function to correct the pointcloud, to make the camera frame is perpendicular to the ground
# ? NOT WORKS
def correct_pc_perpendicular(_pointcloud: np.ndarray, _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1])):
    """
    Correct the pointcloud(make the pointcloud perpendiculart to the ground) based on the camera pose

    Args:
        _pointcloud (np.ndarray): <nums, 3>
        _camera_pose (np.ndarray, optional): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form. Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]).
                                            Note: it should be the original camera pose or 0 # ?

    Returns:
        _pointcloud: <nums, 3>
    """
    _r = R.from_quat(_camera_pose[3:]).as_euler("xyz")
    _r[1:] = 0  # keep the rotation along y
    _R_mat = R.from_euler("xyz", _r).as_matrix()

    _pointcloud = _R_mat.dot(_pointcloud.T).T
    return _pointcloud


#  a test function to draw the 2D pointcloud to check if the function is correct, useless
# Open3d cant see the 2d pointcloud
def test_draw_2Dpointclouds_inblack(
    _pointclouds: np.ndarray,
    _window_name: str = "Pointcloud",
    _background_color: np.ndarray = np.asarray([0, 0, 0]),
    _pointcloud_color: np.ndarray = np.asarray([1, 1, 1]),
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 0]),
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """
    To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to 'Pointcloud'.
        _background_color (np.ndarray, optional): Defaults to np.asarray([0, 0, 0]).
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 0]). The camera pose in the world frame <1, 7> rotaiion vector in quaternion form.
        _pointcloud_color (np.ndarray, optional): Defaults to np.asarray([1, 1, 1]).
        _frame_size (int, optional):  Defaults to 1.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """

    print("Visualize the pointcloud in white:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _vis.get_render_option().background_color = _background_color  # set the color of the background

    _pointclouds_o3d = o3d.geometry.PointCloud()
    print(_pointclouds.shape)

    _pointclouds = np.insert(_pointclouds, 1, 0, axis=1)
    print(_pointclouds)
    print(np.max(_pointclouds))
    print(_pointclouds.shape)

    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_o3d.paint_uniform_color(_pointcloud_color)  # set the color of the pointclouds
    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()
    _vis_controller.set_front((0, 0, -1))
    _vis_controller.set_lookat((1, 0, 0))
    _vis_controller.set_up((0, -1, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# The main function
if __name__ == "__main__":
    # ! Define some GLOBAL parameters
    INITIAL_DIR = "/home/ke/data/tartanair-release1/abandonedfactory/abandonedfactory/Easy/P006"

    Fx = 320.0  # focal length x
    Fy = 320.0  # focal length y
    Cx = 320.0  # optical center x
    Cy = 240.0  # optical center y

    FOV = 90  # field of view /deg

    WIDTH = 640
    HEIGHT = 480

    CAMERA_PARAS = np.array([Fx, Fy, Cx, Cy])

    CAMERA_MAT = np.array([[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]])

    CAMERA_MAT_INV = np.linalg.inv(CAMERA_MAT)

    # ! Load the data from the tartanair dataset
    imgs_l, depths_l, poses_l = tartan_data_loader(INITIAL_DIR, "left", 300, 320)  # the 300 has a good depth image
    # <nums, 7> rotaiion vector in quaternion form, and we keep it during the whole process
    # left_t, left_R_quaternion = poses[:, :3], poses[:, 3:]
    # test = R.from_quat(poses[0, 3:]).as_matrix()

    # ! Modify te camera pose to the ideal frame
    poses_l_ry = remove_rotation_xz(poses_l)  # remove the rotation at first
    poses_l_ry_p0 = move_pose(poses_l_ry, poses_l_ry[0])

    poses_l_p0 = move_pose(poses_l, poses_l[0])  # move the world frame to the first camera pose
    poses_l_p0_ry = remove_rotation_xz(poses_l_p0)

    # ! Which one we want to check
    n = 2
    n_t = 5

    pc_range = np.asarray((15.0, 20.0))
    pc_shift = 10.0
    BEV_size = np.asarray([240, 320])
    N_slice = 1
    i = 0

    # No.5 is good to check the BEV result
    # some depths too big
    # No.10 to No.15 doesnt work maybe not a big problem
    # No.60 around is similar cause the sky
    # ? The No.325 the frame is inverse
    # The reason of the remove of the rotation

    # print("origin pose:", poses_l[n], "\n")
    # print('move to pose 0:', poses_l_p0[n], '\n') # why it modifiy the whole list instead of the copy
    # print('Remove rotation:', poses_l_p0_ry[n], '\n')
    # print('Remove rotation in euler', R.from_quat(poses_l_p0_ry[n, 3:]).as_euler('xyz'))

    # print("Remove rotation:", poses_l_ry[n], "\n")
    # print('Remove rotation in euler', R.from_quat(poses_l_ry[n, 3:]).as_euler('xyz'))
    # print("move to pose 0:", poses_l_ry_p0[n], "\n")  # why it modifiy the whole list instead of the copy

    # ! Plot the images
    # depths_l[0].all() > 0 -> true
    # draw_depth_color(depths_l[n], imgs_l[n])  # 80/depth = disparity

    # I think there is no need to use a big size, the image performance is not good and time consuming O(n)
    # get_BEV_from_depthandcolor(depths_l[n], imgs_l[n], poses_l_p0[n], CAMERA_PARAS,_size = [240, 320] ,_threshold = 30, _N_slice = 1)
    # get_BEV_from_depthandcolor(depths_l[n], imgs_l[n], poses_l_p0[n], CAMERA_PARAS,_size = [240, 320] ,_threshold = 30, _N_slice = 10)

    # get_frontview_from_depthandcolor(depths_l[n], imgs_l[n], poses_l_p0[n], CAMERA_PARAS, _size=[240, 320], _threshold=30, _N_slice=1)
    # get_sideview_from_depthandcolor(depths_l[n], imgs_l[n], poses_l_p0[n], CAMERA_PARAS, _size=[240, 320], _threshold=30, _N_slice=1)

    # ! Get the BEV from the depth and color image
    image_0 = get_BEV_from_depthandcolor(
        depths_l[n],
        imgs_l[n],
        poses_l_p0[n],
        CAMERA_PARAS,
        BEV_size=BEV_size,
        threshold=30,
        n_slice=N_slice,
        pc_range=pc_range,
        pc_shift=pc_shift,
    )

    image_1 = get_BEV_from_depthandcolor(
        depths_l[n + n_t],
        imgs_l[n + n_t],
        poses_l_p0[n + n_t],
        CAMERA_PARAS,
        BEV_size=BEV_size,
        threshold=30,
        n_slice=N_slice,
        pc_range=pc_range,
        pc_shift=pc_shift,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image_0[i] / 255.0)

    plt.subplot(1, 2, 2)
    plt.imshow(image_1[i] / 255.0)
    plt.show()

    print("pose t0:", poses_l_ry_p0[n], "\n")
    print("pose t1:", poses_l_ry_p0[n + n_t], "\n")

    # The ground truth of the pose change
    test_wTi0 = get_T_from_pose(poses_l_ry_p0[n])
    test_i0Tw = inverse_transformation(test_wTi0)

    real_t = np.dot(test_i0Tw, homogenize(poses_l_ry_p0[n + n_t, :3].reshape(-1, 3)).T)[:3, 0]
    real_r = quaternion_multiply(poses_l_ry_p0[n + n_t, 3:], quaternion_inverse(poses_l_ry_p0[n, 3:]))
    groud_truth = np.asarray([real_t[0], real_t[1], real_r[1]])
    print("Ground truth(t, r):", groud_truth)

    # Can not add and minus directly
    # real_t = poses_l_ry_p0[n + n_t, (0, 2)] - poses_l_ry_p0[n, (0, 2)]
    # real_r = R.from_quat(poses_l_ry_p0[n + n_t, 3:]).as_euler("xyz") - R.from_quat(poses_l_ry_p0[n, 3:]).as_euler(
    #     "xyz"
    # )
    # groud_truth = np.asarray([real_t[0], real_t[1], real_r[1]])
    # print("Ground truth(tx, tz,ry):", groud_truth)

    # ! Use a custom ICP function to check the performance
    # Reconstruct the 2d pointcloud from 2 BEV image with their ORB features # ! Feature based method
    pc_2D_0, pc_2D_1 = ORB_BEV_to_2dpc(image_0[i], image_1[i], pc_shape=pc_range / BEV_size, pc_shift=pc_shift)

    # # Reconstruct the 2d pointcloud from the BEV image # ! Classical ICP method, NOT GOOD, maybe it is not good for the dense pointcloud
    # pc_2D_0 = BEV_to_2dpc(image_0[i], pc_shape=pc_range / BEV_size, pc_shift=pc_shift)
    # pc_2D_1 = BEV_to_2dpc(image_1[i], pc_shape=pc_range / BEV_size, pc_shift=pc_shift)

    transformation_history, aligned_points = icp(
        pc_2D_0,
        pc_2D_1,
        max_iterations=10000000,
        distance_threshold=20,
        convergence_translation_threshold=0.0000001,
        convergence_rotation_threshold=0.000001,
        info=False,
    )

    res_icp = -transformation_history[-1]
    print("ICP result:", res_icp)  # return as (t, r)

    # show results
    plt.plot(pc_2D_0[:, 0], pc_2D_0[:, 1], "rx", label="reference points")
    plt.plot(pc_2D_1[:, 0], pc_2D_1[:, 1], "g.", label=" target points")
    plt.plot(aligned_points[:, 0], aligned_points[:, 1], "b+", label="aligned points")
    plt.legend()
    plt.show()

    pose = res_icp
    Tran = np.eye(3)
    Tran[0, -1] = pose[0]
    Tran[1, -1] = pose[1]
    Tran[:2, :2] = np.array([[math.cos(pose[2]), -math.sin(pose[2])], [math.sin(pose[2]), math.cos(pose[2])]])

    real_pc_2D_0 = BEV_to_2dpc(image_0[i], pc_shape=pc_range / BEV_size, pc_shift=pc_shift)
    real_pc_2D_1 = BEV_to_2dpc(image_1[i], pc_shape=pc_range / BEV_size, pc_shift=pc_shift)

    real_pc_2D = Tran.dot(homogenize(real_pc_2D_1).T).T[:, :2]

    plt.plot(real_pc_2D_0[:, 0], real_pc_2D_0[:, 1], "rx", label="reference points")
    plt.plot(real_pc_2D_1[:, 0], real_pc_2D_1[:, 1], "g.", label=" target points")
    plt.plot(real_pc_2D[:, 0], real_pc_2D[:, 1], "b+", label="aligned points")
    plt.legend()
    plt.show()

    # ! Convert 2d pointcloud to the 3d pointcloud to use open3d ICP to check the performance
    # pc_3D_0 = np.insert(pc_2D_0, 1, 0, axis=1)
    # pc_3D_1 = np.insert(pc_2D_1, 1, 0, axis=1)

    # pc_3D_0_o3d = o3d.geometry.PointCloud()
    # pc_3D_0_o3d.points = o3d.utility.Vector3dVector(pc_3D_0)

    # pc_3D_1_o3d = o3d.geometry.PointCloud()
    # pc_3D_1_o3d.points = o3d.utility.Vector3dVector(pc_3D_1)

    # trans_init = np.asarray(
    #     [[0.862, 0.011, -0.507, 0.5], [-0.139, 0.967, -0.215, 0.7], [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    # )

    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pc_3D_0_o3d,
    #     pc_3D_1_o3d,
    #     max_correspondence_distance=0.1,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # )
    # print("Transformation is:", reg_p2p.transformation)

    # print(pc_2D_0.shape)
    # # print(set(tuple([tuple(i) for i in pc_2D_0])))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(pc_2D_0[:, 0], pc_2D_0[:, 1], s=2, c="k")
    # plt.show()

    # plt.subplot(1, 2, 1)
    # plt.imshow(pc_2D_0, cmap="gray")

    # plt.subplot(1, 2, 2)
    # plt.imshow(pc_2D_1, cmap="gray")
    # plt.show()

    # ! Visualize the pointcloud to debug
    pointcloud = pointcloud_from_depth(
        depths_l[n], CAMERA_PARAS, threshold=20
    )  # generate the pointcloud from the depth img

    pointcloud_w = camera2world(
        pointcloud, poses_l_p0[n]
    )  # change the pointcloud from the camera frame to the world frame

    # What is the pointcloud if the camera just rotate along the y axis
    pointcloud_ry = correct_pc_rotationxz(pointcloud, poses_l_p0[n])
    pointcloud_ry = camera2world(pointcloud_ry, poses_l_p0[n])
    # pointcloud_p = correct_pc_perpendicular(pointcloud, poses_l[n])

    # Visualize the pointcloud in the world frame
    # draw_pointclouds_inblack(pointcloud, _window_name = 'depth', _camera_pose = poses_l_p0[n]) # Draw it without correcting
    draw_pointclouds_incolor(
        pointcloud_w, imgs_l[n], _window_name="color", _frame_size=0.7, _point_size=4, _camera_pose=poses_l_p0_ry[n]
    )

    # test_draw_pointclouds(
    #     pointcloud_w,
    #     pointcloud_ry,
    #     imgs_l[n],
    #     _window_name="color",
    #     _frame_size=0.7,
    #     _point_size=4,
    #     _camera_pose=poses_l_p0[n],
    # )
