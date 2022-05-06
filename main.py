# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pylie import SO3, SE3

from PerspectiveCamera import PerspectiveCamera
from common_lab_utils import LocalCoordinateSystem, GeodeticPosition, Attitude, CartesianPosition, homogeneous, hnormalized
from dataset import getPosesFromData
from dataset2 import Dataset

def to_SO3(attitude: Attitude):
    """Convert to SO3 representation."""


    return SO3.rot_z(attitude.z_rot) @ SO3.rot_y(attitude.y_rot) @ SO3.rot_x(attitude.x_rot)

def to_vector(position: CartesianPosition):
    """Convert to column vector representation."""

    return np.array([[position.x, position.y, position.z]]).T

def zncc_test(img1, img2, W):

    zncc_image = np.zeros(img1.shape)
    for i in range(W, img1.shape[0] - W):
        for j in range(W, img1.shape[1] - W):
            subimg1 = img1[i - W:i + W, j - W:j + W]

            subimg2 = img2[i - W:i + W, j - W:j + W]

            mu1, mu2 = np.mean(subimg1), np.mean(subimg2)
            sigma1, sigma2 = np.std(subimg1), np.std(subimg2)
            if np.abs(sigma1) > 1e-8 and np.abs(sigma2) > 1e-8:


                subImg1Norm = subimg1 - mu1
                subImg2Norm = subimg2 - mu2

                flatimg1 = subImg1Norm.flatten()
                flatimg2 = subImg2Norm.flatten()

                # np.linalg.norm(subImg1Norm)

                zncc_image[i, j] = (subImg1Norm.flatten().T @ subImg2Norm.flatten())/(sigma1*sigma2*(2*W+1)**2)
                #zncc_image[i-W:i+W, j-W:j+W] = np.inner(flatimg1/np.linalg.norm(flatimg1), flatimg2/np.linalg.norm(flatimg2))

                #print(np.sum(zncc_image_explicit - zncc_image))

    return zncc_image

def getPoses(image_series, scale, local_system):

    HEIGHT, WIDTH = image_series[0].image.shape

    position_geodetic = image_series[-1].body_position_in_geo
    orientation_ned_body = to_SO3(image_series[-1].body_attitude_in_geo)

    # Compute the pose of the body in the local coordinate system.
    pose_local_body = local_system.to_local_pose(position_geodetic, orientation_ned_body)

    position_body_camera = to_vector(image_series[-1].camera_position_in_body)
    orientation_body_camera = to_SO3(image_series[-1].camera_attitude_in_body)
    pose_body_camera = SE3((orientation_body_camera, position_body_camera))
    pose_local_camera_ref = pose_local_body @ pose_body_camera
    relative_T = []
    undistorted_image_series = []

    for element in image_series:
        position_geodetic = element.body_position_in_geo
        orientation_ned_body = to_SO3(element.body_attitude_in_geo)

        # Compute the pose of the body in the local coordinate system.
        pose_local_body = local_system.to_local_pose(position_geodetic, orientation_ned_body)
        position_body_camera = to_vector(element.camera_position_in_body)
        orientation_body_camera = to_SO3(element.camera_attitude_in_body)
        pose_body_camera = SE3((orientation_body_camera, position_body_camera))
        pose_local_camera = pose_local_body @ pose_body_camera
        pose_cameraref_camera = (pose_local_camera_ref.inverse() @ pose_local_camera)
        relative_T.append(pose_cameraref_camera.inverse())

        camera_model = PerspectiveCamera.from_intrinsics(element.intrinsics, pose_local_camera)

        undistorted_img = camera_model.undistort_image(element.image)

        undistorted_image_series.append(cv2.resize(undistorted_img, (round(WIDTH * scale), round(HEIGHT * scale))))

    relative_R = []
    relative_t = []

    for i in range(3):
        relative_R.append(relative_T[i].rotation.matrix)
        relative_t.append(relative_T[i].translation)

    return relative_R, relative_t, undistorted_image_series, pose_local_camera_ref



def match(img1, img2, W = 7, alpha = 0.6):

    subimgtensor1 = np.zeros((img1.shape[0], img1.shape[1], 2*W, 2*W))
    subimgtensor2 = np.zeros((img2.shape[0], img2.shape[1], 2 * W, 2 * W))

    for i in range(W, img1.shape[0]-W):
        for j in range(W, img1.shape[1]-W):
            subimgtensor1[i, j] = img1[i-W:i+W, j-W:j+W]
            subimgtensor2[i, j] = img2[i-W:i+W, j-W:j+W]

    mu1_image = np.mean(subimgtensor1, axis=(2,3))
    mu2_image = np.mean(subimgtensor2, axis=(2, 3))

    sigma1 = np.std(subimgtensor1, axis=(2,3))
    sigma2 = np.std(subimgtensor2, axis=(2, 3))

    subimgtensor1 = subimgtensor1 - mu1_image[:, :, np.newaxis, np.newaxis]
    subimgtensor2 = subimgtensor2 - mu2_image[:, :, np.newaxis, np.newaxis]

    reduced1 = np.divide(subimgtensor1, sigma1[:, :, np.newaxis, np.newaxis])
    reduced2 = np.divide(subimgtensor2, sigma2[:, :, np.newaxis, np.newaxis])

    zncc_image = np.sum(reduced1 * reduced2, axis=(2,3))  # omitting this for efficiency: /((2*W+1)**2)

    zncc_quality_mask = zncc_image < alpha * (2*W+1)**2 # removing low quality matches
    zncc_valid_mask = ~np.isfinite(zncc_image) # # removing nans and infinities
    zncc_mask = zncc_quality_mask + zncc_valid_mask
    zncc_image[zncc_mask] = 0

    return zncc_image

def getDepth(image_series, local_system, scale=0.5, d_range = (500, 1000, 10), alpha = 0.7):


    S = np.diag([scale, scale, 1])
    # S = np.identity(3)

    HEIGHT, WIDTH = image_series[0].image.shape
    n = np.array([0, 0, -1])

    K_ref = S @ np.array(
        [[image_series[-1].intrinsics.fu, image_series[-1].intrinsics.s, image_series[-1].intrinsics.cu],
         [0, image_series[-1].intrinsics.fv, image_series[-1].intrinsics.cv],
         [0, 0, 1]])

    K_ref_inv = np.linalg.inv(K_ref)

    relative_R, relative_t, undistorted_image_series, pose_local_camera_ref = getPoses(image_series, scale, local_system)

    d_list = [e for e in range(d_range[0], d_range[1], d_range[2])]

    M_arr = np.zeros((round(HEIGHT * scale), round(WIDTH * scale), len(d_list)))

    for j, d in enumerate(d_list):
        zncc_img_list = []
        for i in range(3):
            K = S @ np.array(
                [[image_series[i].intrinsics.fu, image_series[i].intrinsics.s, image_series[i].intrinsics.cu],
                 [0, image_series[i].intrinsics.fv, image_series[i].intrinsics.cv],
                 [0, 0, 1]])

            H = K @ (relative_R[i] - np.outer(relative_t[i].T, n) / d) @ K_ref_inv
            H_inv = np.linalg.inv(H)

            IkWarped = cv2.warpPerspective(undistorted_image_series[i], H_inv,
                                           (round(WIDTH * scale), round(HEIGHT * scale)))
            zncc_img_list.append(match(IkWarped, undistorted_image_series[-1], alpha=alpha))


        zncc_sum = sum(zncc_img_list)

        M_arr[:, :, j] = zncc_sum

    depth_image = (np.argmax(M_arr, axis=2) * d_range[2] + d_range[0])

    points = []
    colors = []
    print(depth_image.shape)
    for m in range(depth_image.shape[0]):
        for n in range(depth_image.shape[1]):
            if depth_image[m, n] > 500:
                X = (n - scale*image_series[-1].intrinsics.cu) / (scale*image_series[-1].intrinsics.fu) * depth_image[m, n]
                Y = (m - scale*image_series[-1].intrinsics.cv) / (scale*image_series[-1].intrinsics.fv) * depth_image[m, n]
                point_camera = np.array([X, Y, depth_image[m, n]])[:, np.newaxis]
                point_local = hnormalized(pose_local_camera_ref.to_matrix() @ homogeneous(point_camera))
                points.append(point_local.flatten())
                colors.append([undistorted_image_series[-1][m, n], undistorted_image_series[-1][m, n], undistorted_image_series[-1][m, n]])


    return depth_image, undistorted_image_series[-1], points, colors, pose_local_camera_ref

if __name__ == '__main__':

    dataset = Dataset()

    image_series = []
    scale = 0.5

    local_system = LocalCoordinateSystem(GeodeticPosition(59.963516, 10.667307, 321.0))

    vis = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()
    vis.create_window("pointcloud")
    vis2.create_window("depth image")
    pcd = o3d.geometry.PointCloud()

    for i, e in enumerate(dataset):
        if i < 2:
            image_series.append(e)
        else:

            image_series.append(e)
            depth_image, undist_image, points, colors, pose_local_camera = getDepth(image_series, local_system, scale, d_range=(500, 1200, 20), alpha=0.8)
            """plt.figure()
            plt.imshow(depth_image, cmap="gray")
            plt.title(f"Image number {i}")
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.001)"""

            image_series.pop(0)

            points_arr = np.array(points)

            img = o3d.geometry.Image(depth_image.astype(np.uint16))

            intr = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[0], depth_image.shape[1], scale*image_series[-1].intrinsics.fu, scale*image_series[-1].intrinsics.fv, scale*image_series[-1].intrinsics.cu, scale*image_series[-1].intrinsics.cv)

            #pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intr)

            pcd.points.extend(o3d.utility.Vector3dVector(points_arr))
               # = o3d.utility.Vector3dVector(points_arr)
            pcd.colors.extend(o3d.utility.Vector3dVector(np.array(colors)/255))
            pcd = pcd.voxel_down_sample(1)
            #o3d.visualization.draw_geometries([pcd])

            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()


            vis2.clear_geometries()
            vis2.add_geometry(img)
            vis2.poll_events()
            vis2.update_renderer()


    while True:
        vis.poll_events()
        vis.update_renderer()



