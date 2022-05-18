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
from dataset_kimera import DatasetKimera


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
def getPosesKimera(image_series, scale):
    HEIGHT, WIDTH = image_series[0].image.shape

    pose_local_camera_ref = image_series[-1].pose
    relative_T = []
    undist_img_series = []

    for element in image_series:
        pose_local_camera = element.pose
        pose_cameraref_camera = (pose_local_camera_ref.inverse() @ pose_local_camera)
        relative_T.append(pose_cameraref_camera.inverse())

        undist_img_series.append(cv2.resize(element.image, (round(WIDTH * scale), round(HEIGHT * scale))))


    relative_R = []
    relative_t = []

    for i in range(3):
        relative_R.append(relative_T[i].rotation.matrix)
        relative_t.append(relative_T[i].translation)

    return relative_R, relative_t, undist_img_series, pose_local_camera_ref


def getPosesKollen(image_series, scale, local_system):

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
    vis.poll_events()
    vis.update_renderer()

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

def getDepth(image_series, local_system, relative_R, relative_t, undistorted_image_series, pose_local_camera_ref, scale=0.5, d_range = (500, 1000, 10), alpha = 0.7):


    S = np.diag([scale, scale, 1])
    # S = np.identity(3)

    HEIGHT, WIDTH = image_series[0].image.shape
    n = np.array([0, 0, -1])

    """K_ref = S @ np.array(
        [[image_series[-1].intrinsics.fu, image_series[-1].intrinsics.s, image_series[-1].intrinsics.cu],
         [0, image_series[-1].intrinsics.fv, image_series[-1].intrinsics.cv],
         [0, 0, 1]])"""

    K_ref = S @ np.array(
        [[415.692193817, 0, 360],
         [0, 415.692193817, 240],
         [0, 0, 1]])


    K_ref_inv = np.linalg.inv(K_ref)

    #relative_R, relative_t, undistorted_image_series, pose_local_camera_ref = getPosesKollen(image_series, scale, local_system)

    d_list = np.linspace(d_range[0], d_range[1], d_range[2])
    M_arr = np.zeros((round(HEIGHT * scale), round(WIDTH * scale), len(d_list)))

    img_sum = np.zeros((round(HEIGHT * scale), round(WIDTH * scale), 3))
    img_sum[:,:, 2] = cv2.resize(image_series[-1].image, (round(WIDTH * scale), round(HEIGHT * scale)))/255

    sweep_images = []

    for j, d in enumerate(d_list):
        zncc_img_list = []
        for i in range(2):
            """K = S @ np.array(
                [[image_series[i].intrinsics.fu, image_series[i].intrinsics.s, image_series[i].intrinsics.cu],
                 [0, image_series[i].intrinsics.fv, image_series[i].intrinsics.cv],
                 [0, 0, 1]])"""

            K = S @ np.array(
                [[415.692193817, 0, 360],
                 [0, 415.692193817, 240],
                 [0, 0, 1]])



            H = K @ (relative_R[i] - np.outer(relative_t[i].T, n) / d) @ K_ref_inv
            H_inv = np.linalg.inv(H)

            IkWarped = cv2.warpPerspective(undistorted_image_series[i], H_inv,
                                           (round(WIDTH * scale), round(HEIGHT * scale)))
            zncc_img_list.append(match(IkWarped, undistorted_image_series[-1], alpha=alpha))
            img_sum[:, :, i] = IkWarped/255


        zncc_sum = sum(zncc_img_list)
        sweep_images.append(img_sum.copy())
        img = o3d.geometry.Image((img_sum*255).astype(np.uint8))
        vis2.clear_geometries()
        vis2.add_geometry(img)
        vis2.poll_events()
        vis2.update_renderer()
        M_arr[:, :, j] = zncc_sum

    depth_image = (np.argmax(M_arr, axis=2) *(d_range[1]- d_range[0])/d_range[2] + d_range[0])


    points = []
    colors = []
    for m in range(depth_image.shape[0]):
        for n in range(depth_image.shape[1]):
            if depth_image[m, n] > d_range[0]:
                """X = (n - scale*image_series[-1].intrinsics.cu) / (scale*image_series[-1].intrinsics.fu) * depth_image[m, n]
                Y = (m - scale*image_series[-1].intrinsics.cv) / (scale*image_series[-1].intrinsics.fv) * depth_image[m, n]"""

                X = (n - scale * 360) / (scale * 415.692193817) * \
                    depth_image[m, n]
                Y = (m - scale * 240) / (scale * 415.692193817) * \
                    depth_image[m, n]

                point_camera = np.array([X, Y, depth_image[m, n]])[:, np.newaxis]

                point_local = point_camera # hnormalized(image_series[-1].pose.to_matrix() @ homogeneous(point_camera))

                points.append(point_local.flatten())

                m_s, n_s = round(m / scale), round(n / scale)

                colors.append([image_series[-1].image_segmented[m_s, n_s, 2], image_series[-1].image_segmented[m_s, n_s, 1],
                               image_series[-1].image_segmented[m_s, n_s, 0]])

                """colors.append([image_series[-1].image_bgr[m_s, n_s, 2], image_series[-1].image_bgr[m_s, n_s, 1],
                   image_series[-1].image_bgr[m_s, n_s, 0]])"""

    return depth_image, undistorted_image_series[-1], points, colors, pose_local_camera_ref, sweep_images

if __name__ == '__main__':

    dataset = DatasetKimera()

    image_series = []
    scale = 0.5

    local_system = LocalCoordinateSystem(GeodeticPosition(59.963516, 10.667307, 321.0))

    d_kollen = (500, 1200, 30)
    d_kimera = (0.1, 12, 50)

    vis = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()
    vis.create_window("pointcloud")
    vis2.create_window("depth image")
    pcd = o3d.geometry.PointCloud()

    sweep_counter = 0
    depth_counter = 0

    for i, e in enumerate(dataset):
        if i < 2:
            image_series.append(e)



        else:

            """        elif i == 200:
            break"""



            image_series.append(e)

            relative_R, relative_t, undistorted_image_series, pose_local_camera_ref = getPosesKimera(image_series,
                                                                                                     scale)

            """relative_R, relative_t, undistorted_image_series, pose_local_camera_ref = getPosesKollen(image_series,
                                                                                                     scale, local_system)"""
            # checking if there's been any significant movement:
            if (np.linalg.norm(relative_R[0] - relative_R[2]) > 1 or
                np.linalg.norm(relative_R[1] - relative_R[2]) > 1) or \
                (np.linalg.norm(relative_t[0] - relative_t[2]) > 1 or
                 np.linalg.norm(relative_t[1] - relative_t[2]) > 1):

                depth_image, undist_image, points, colors, pose_local_camera, sweep_images = getDepth(image_series, local_system, relative_R, relative_t, undistorted_image_series, pose_local_camera_ref,
                                                                                        scale, d_range=d_kimera,
                                                                                        alpha=0.8)

                for e in sweep_images:
                    cv2.imwrite("sweep_out/" + str(sweep_counter).zfill(6) + ".jpg", e*255)
                    sweep_counter += 1

                print(np.max(depth_image))
                cv2.imwrite("depth_out/" + str(depth_counter).zfill(6) + ".jpg", depth_image/12*255)
                depth_counter += 1

                img = o3d.geometry.Image(depth_image.astype(np.uint16))

                # intr = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[0], depth_image.shape[1], scale*image_series[-1].intrinsics.fu, scale*image_series[-1].intrinsics.fv, scale*image_series[-1].intrinsics.cu, scale*image_series[-1].intrinsics.cv)

                # pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intr)

                if len(points) > 0:
                    points_arr = np.array(points)
                    print("pose:",pose_local_camera_ref)

                    new_points = o3d.utility.Vector3dVector(points_arr)
                    new_pcd = o3d.geometry.PointCloud(new_points)
                    new_pcd.transform(pose_local_camera_ref.to_matrix())
                    new_pcd.colors.extend(np.array(colors) / 255)

                    new_pcd, _ = new_pcd.remove_statistical_outlier(10, 4)


                    #pcd.points.extend(o3d.utility.Vector3dVector(points_arr))
                    pcd = pcd + new_pcd
                    # = o3d.utility.Vector3dVector(points_arr)
                    #pcd.colors.extend(o3d.utility.Vector3dVector(np.array(colors) / 255))

                    pcd = pcd.voxel_down_sample(voxel_size=0.2)
                # o3d.visualization.draw_geometries([pcd])

                vis.clear_geometries()
                vis.add_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                img = o3d.geometry.Image(depth_image.astype(np.uint16))
                vis2.clear_geometries()
                vis2.add_geometry(img)
                vis2.poll_events()
                vis2.update_renderer()

                image_series.pop(0)

            else:
                image_series.pop()

    o3d.io.write_point_cloud("kimera_pcd_2.pcd", pcd)


    while True:
        vis.poll_events()
        vis.update_renderer()



