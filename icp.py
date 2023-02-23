import math
import numpy as np
from sklearn.neighbors import NearestNeighbors


def point_based_matching(point_pairs: np.ndarray):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.
    Args:
        point_pairs (np.ndarray): <nums, 4> (xi, yi, xpi, ypi)

    Returns:
        rot_angle, translation_x, translation_y, the best transformation under this situation
    """

    n = len(point_pairs)
    if n == 0:
        return None, None, None

    x_mean, y_mean, xp_mean, yp_mean = np.mean(point_pairs, axis=0)

    s_x, s_y, s_xp, s_yp = np.hsplit(point_pairs - np.mean(point_pairs, axis=0), 4)
    s_x_xp = np.sum(s_x * s_xp)
    s_y_yp = np.sum(s_y * s_yp)
    s_x_yp = np.sum(s_x * s_yp)
    s_y_xp = np.sum(s_y * s_xp)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
    translation_y = yp_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(
    reference_points,
    points,
    max_iterations=100,
    distance_threshold=0.3,
    convergence_translation_threshold=1e-3,
    convergence_rotation_threshold=1e-4,
    point_pairs_threshold=10,
    info=False,
):
    """An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.

    Args:
        reference_points (_type_): the reference point set as a numpy array (N x 2)
        points (_type_): the point that should be aligned to the reference_points set as a numpy array (M x 2)
        max_iterations (int, optional): the maximum number of iteration to be executed
        distance_threshold (float, optional): the distance threshold between two points in order to be considered as a
        pair
        convergence_translation_threshold (_type_, optional): the threshold for the translation parameters (x and y)
        for the transformation to be considered converged
        convergence_rotation_threshold (_type_, optional): _the threshold for the rotation angle (in rad) for the
        transformation to be considered converged
        point_pairs_threshold (int, optional): the minimum number of point pairs the should exist
        info (bool, optional): whether to print informative messages about the process (default: False)
    Returns:
        _type_: the transformation history as a list of numpy arrays containing the (tx, ty, r)
    """

    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(reference_points)

    for iter_num in range(max_iterations):
        if info:
            print("------ iteration", iter_num, "------")

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append(
                    (
                        points[nn_index][0],
                        points[nn_index][1],
                        reference_points[indices[nn_index][0]][0],
                        reference_points[indices[nn_index][0]][1],
                    )
                )

        # if only few point pairs, stop process
        if info:
            print("number of pairs found:", len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if info:
                print("No better solution can be found (very few point pairs)!")
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(
            np.asarray(closest_point_pairs)
        )
        if closest_rot_angle is not None:
            if info:
                print("Rotation:", math.degrees(closest_rot_angle), "degrees")
                print("Translation:", closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if info:
                print("No better solution can be found!")
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s], [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.asarray([closest_translation_x, closest_translation_y, closest_rot_angle]))

        # check convergence
        if (
            (abs(closest_rot_angle) < convergence_rotation_threshold)
            and (abs(closest_translation_x) < convergence_translation_threshold)
            and (abs(closest_translation_y) < convergence_translation_threshold)
        ):
            if info:
                print("Converged!")
            break

    return transformation_history, points
