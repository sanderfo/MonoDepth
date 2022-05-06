from pylie import SO3, SE3
import numpy as np

def getPosesFromData(img_n):
    pose_bundlerc_c = SE3((SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])), np.zeros([3, 1])))
    with open("data/Holmenkollendatasett_plane.out", "r") as file:
        line = file.readline()
        if not line.startswith("# Bundle file v0.3"):
            print("Warning: Expected v0.3 Bundle file, but first line did not match the convention")

        num_cameras, num_points = [int(x) for x in next(file).split()]

        R_list = []
        t_list = []
        T_list = []


        for i in range(3):
            _f, k1, k2 = [float(x) for x in next(file).split()]

            if k1 != 0 or k2 != 0:
                print("The current implementation assumes undistorted data. Distortion parameters are ignored")

            # Read the rotation matrix from the next three lines.
            R = np.array([[float(x) for x in next(file).split()] for y in range(3)]).reshape([3, 3])

            # Read the translation from the last line.
            t = np.array([float(x) for x in next(file).split()]).reshape([3, 1])

            R_list.append(R)
            t_list.append(t)

            T_list.append(SE3((SO3(R), t)).inverse() @ pose_bundlerc_c)

            # T is given as T_c_w

        return T_list