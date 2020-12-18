from utils.functions import draw_landmarks, get_suffix, parse_roi_box_from_landmark

from scipy.io import loadmat
from math import cos, sin, atan2, asin, sqrt
import numpy as np
import cv2


def matrix2angle(R):
  """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
  if R[2, 0] > 0.998:
    z = 0
    x = np.pi / 2
    y = z + atan2(-R[0, 1], -R[0, 2])
  elif R[2, 0] < -0.998:
    z = 0
    x = -np.pi / 2
    y = -z + atan2(R[0, 1], R[0, 2])
  else:
    x = asin(R[2, 0])
    y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
    z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

  return x, y, z


def angle2matrix(phi, gamma, theta):
  """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
  refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
  todo: check and debug
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw    thetha
        y: pitch  gamma
        z: roll   phi
    """

  # theta = -yaw
  # psi = pitch
  # phi = roll

  # R = np.ndarray(shape=(3, 3))
  # R[0, 0] = cos(theta)*cos(phi)
  # R[0, 1] = sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)
  # R[0, 2] = cos(psi)*sin(theta)*cos(phi) - sin(psi)*sin(phi)
  # R[1, 0] = cos(theta)*sin(phi)
  # R[1, 1] = sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi)
  # R[1, 2] = cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)
  # R[2, 0] = -sin(theta)
  # R[2, 1] = sin(psi)*cos(theta)
  # R[2, 2] = cos(psi)*cos(theta)

  # Porting from matlab codes
  R_x = np.array([[1, 0, 0], [0, cos(phi), sin(phi)], [0, -sin(phi), cos(phi)]])
  R_y = np.array([[cos(gamma), 0, -sin(gamma)], [0, 1, 0],
                  [sin(gamma), 0, cos(gamma)]])
  R_z = np.array([[cos(theta), sin(theta), 0], [-sin(theta),
                                                cos(theta), 0], [0, 0, 1]])

  R = np.matmul(np.matmul(R_x, R_y), R_z)

  # To be compatible with 3DFFA
  R = R * [[1], [-1], [1]]

  return R


def load_image_and_its_params(img_path, mat_path):
  img = cv2.imread(img_path)
  assert (img.shape[0] == img.shape[1])
  data = loadmat(mat_path)

  # Load 40 shape parameters
  shape_params = data['Shape_Para'][:40]
  # Load 10 exp parameters
  exp_params = data['Exp_Para'][:10]

  # Load 12 translation parameters
  P = np.ndarray(shape=(3, 4))
  P[:, :3] = angle2matrix(data['Pose_Para'][0][0], data['Pose_Para'][0][1],
                          data['Pose_Para'][0][2])
  P[0, 3] = data['Pose_Para'][0][3]
  P[1, 3] = img.shape[1] - data['Pose_Para'][0][4]
  P[2, 3] = data['Pose_Para'][0][5]

  # Put together P, shape and exp params
  img_params = np.array(
      list(P.reshape(-1, 1)) + list(shape_params) + list(exp_params))

  scale = data['Pose_Para'][0][6]

  return img, img_params, scale


if __name__ == "__main__":

  # Load image data
  img, img_params = load_image_and_its_params(
      '/home/innovplus/Dream/Projects/Data/AFLW2000/images/image00002.jpg',
      '/home/innovplus/Dream/Projects/Data/AFLW2000/images/image00002.mat')

  print(img_params.shape)
