import numpy as np
from pylie import SE3
from common_lab_utils import Intrinsics, hnormalized, homogeneous
import cv2


class PerspectiveCamera:
    """Camera model for the perspective camera"""

    def __init__(self,
                 calibration_matrix: np.ndarray,
                 distortion_coeffs: np.ndarray,
                 pose_world_camera: SE3):
        """Constructs the camera model.

        :param calibration_matrix: The intrinsic calibration matrix.
        :param distortion_coeffs: Distortion coefficients on the form [k1, k2, p1, p2, k3].
        :param pose_world_camera: The pose of the camera in the world coordinate system.
        """
        self._calibration_matrix = calibration_matrix
        self._calibration_matrix_inv = np.linalg.inv(calibration_matrix)
        self._distortion_coeffs = distortion_coeffs
        self._pose_world_camera = pose_world_camera
        self._camera_projection_matrix = self._compute_camera_projection_matrix()

    @classmethod
    def from_intrinsics(cls, intrinsics: Intrinsics, pose_world_camera: SE3):
        """Construct a PerspectiveCamera from an Intrinsics object

        :param intrinsics: The camera model intrinsics
        :param pose_world_camera: The pose of the camera in the world coordinate system.
        """

        calibration_matrix = np.array([
            [intrinsics.fu, intrinsics.s,   intrinsics.cu],
            [0.,            intrinsics.fv,  intrinsics.cv],
            [0.,            0.,             1.]
        ])

        distortion_coeffs = np.array([intrinsics.k1, intrinsics.k2, 0., 0., intrinsics.k3])

        return cls(calibration_matrix, distortion_coeffs, pose_world_camera)

    def _compute_camera_projection_matrix(self):
        return self._calibration_matrix @ self._pose_world_camera.inverse().to_matrix()[:3, :]

    def project_world_point(self, point_world):
        """Projects a world point into pixel coordinates.

        :param point_world: A 3D point in world coordinates.
        """

        if point_world.ndim == 1:
            # Convert to column vector.
            point_world = point_world[:, np.newaxis]

        return hnormalized(self._camera_projection_matrix @ homogeneous(point_world))

    def undistort_image(self, distorted_image):
        """Undistorts an image corresponding to the camera model.

        :param distorted_image: The original, distorted image.
        :returns: The undistorted image.
        """

        return cv2.undistort(distorted_image, self._calibration_matrix, self._distortion_coeffs)

    def pixel_to_normalised(self, point_pixel):
        """Transform a pixel coordinate to normalised coordinates

        :param point_pixel: The 2D point in the image given in pixels.
        """

        if point_pixel.ndim == 1:
            # Convert to column vector.
            point_pixel = point_pixel[:, np.newaxis]

        return self._calibration_matrix_inv @ homogeneous(point_pixel)

    @property
    def pose_world_camera(self):
        """The pose of the camera in world coordinates."""
        return self._pose_world_camera

    @property
    def calibration_matrix(self):
        """The intrinsic calibration matrix K."""
        return self._calibration_matrix

    @property
    def calibration_matrix_inv(self):
        """The inverse calibration matrix K^{-1}."""
        return self._calibration_matrix_inv

    @property
    def distortion_coeffs(self):
        """The distortion coefficients on the form [k1, k2, p1, p2, k3]."""
        return self._distortion_coeffs

    @property
    def projection_matrix(self):
        """The projection matrix P."""
        return self._camera_projection_matrix
