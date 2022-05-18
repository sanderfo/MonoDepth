import os
from dataclasses import dataclass
import numpy as np
from pylie import SE3, SO3
import cv2





@dataclass
class KimeraElement:
    """A set of data for each image in the dataset."""
    img_num: int
    image: np.ndarray
    image_segmented: np.ndarray
    pose: SE3


class DatasetKimera:
    """Represents the dataset for this lab."""

    _first_file_num = 0
    _last_file_num = 522
    SE3_list = []

    _curr_file_num = _first_file_num

    def __init__(self, data_dir="data_kimera"):
        self._data_dir = data_dir
        self.readquaternions()
        print("0", self.SE3_list[0])
        print("82", self.SE3_list[82])
        print("92", self.SE3_list[92])
        print(len(self.SE3_list))

    def __iter__(self):
        self._curr_file_num = self._first_file_num
        return self

    def __next__(self):
        """Reads the next data element."""
        if self._curr_file_num > self._last_file_num:
            raise StopIteration
        frame_num = str(self._curr_file_num).zfill(4)
        image_filename = os.path.join(self._data_dir, f"images_gray/frame{frame_num}.jpg")
        image_seg_filename = os.path.join(self._data_dir, f"images_segmented/frame{frame_num}.jpg")

        next_data_element = KimeraElement(
            self._curr_file_num,
            cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE),
            cv2.imread(image_seg_filename),
            self.SE3_list[self._curr_file_num]
        )

        self._curr_file_num += 1
        return next_data_element

    def readquaternions(self, filename="/poses.txt"):
        frame_ids = []
        with open(self._data_dir + "/camerainfo.txt", "r") as file:
            file.readline()
            for line in file:
                values = line.split(",")
                frame_ids.append(values[2])

        frame_counter = 0
        with open(self._data_dir + filename, "r") as file:
            file.readline()  # removing header
            for i, line in enumerate(file):

                values = line.split(",")
                frame_id = values[2]
                if frame_counter < len(frame_ids) and frame_id == frame_ids[frame_counter]:

                    q = [float(values[-1]), float(values[-4]), float(values[-3]), float(values[-2])]
                    t = np.array([values[5], values[6], values[7]], dtype=np.float64)
                    frame_counter += 1
                    self.SE3_list.append(self.quaternion_t_to_SE3(q, t))



    def quaternion_t_to_SE3(self, q, t):
        matrix = np.array(
            [[2 * (q[0] ** 2 + q[1] ** 2) - 1, 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
             [2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[0] ** 2 + q[2] ** 2) - 1, 2 * (q[2] * q[3] - q[0] * q[1])],
             [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 2 * (q[0] ** 2 + q[3] ** 2) - 1]])

        return SE3((SO3(matrix), t))
