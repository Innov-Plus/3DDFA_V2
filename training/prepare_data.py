from utils.functions import draw_landmarks, get_suffix, parse_roi_box_from_landmark
from utils.tddfa_util import _parse_param
from training.export_bfm import load_origininal_bfm

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


class CustomBFMModel(object):

  def __init__(self):
    self.u_base, self.w_shp_base, self.w_exp_base = load_origininal_bfm(
        '/home/innovplus/Dream/Projects/Data/300W_LP/Code/ModelGeneration/Model_Shape.mat',
        '/home/innovplus/Dream/Projects/Data/300W_LP/Code/ModelGeneration/Model_Exp.mat'
    )


class Image3DDFA():

  def __init__(self, img_path, mat_path):
    self.img, self.img_params, self.scale = load_image_and_its_params(
        img_path, mat_path)

  def calculate68Points(self, customBFMModel):
    self.R, self.offset, self.alpha_shp, self.alpha_exp = _parse_param(
        self.img_params)

    self.pts3d = self.scale * self.R @ (
        customBFMModel.u_base + customBFMModel.w_shp_base @ self.alpha_shp +
        customBFMModel.w_exp_base @ self.alpha_exp).reshape(
            3, -1, order='F') + self.offset

  def preprocess(self, customBFMModel):
    # Crop face
    roi_box = parse_roi_box_from_landmark(self.pts3d)

    # self.offset[0] = self.offset[0] - roi_box[0]
    # self.offset[1] = self.offset[1] - roi_box[1]

    self.pts3d = self.scale * self.R @ (
        customBFMModel.u_base + customBFMModel.w_shp_base @ self.alpha_shp +
        customBFMModel.w_exp_base @ self.alpha_exp).reshape(
            3, -1, order='F') + self.offset

    # Resize to 120x120

    return roi_box


if __name__ == "__main__":

  # Load image data
  img, img_params = load_image_and_its_params(
      '/home/innovplus/Dream/Projects/Data/AFLW2000/images/image00002.jpg',
      '/home/innovplus/Dream/Projects/Data/AFLW2000/images/image00002.mat')

  print(img_params.shape)
