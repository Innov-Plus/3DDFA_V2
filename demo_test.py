from training.prepare_data import *
from training.export_bfm import *
from utils.tddfa_util import _parse_param, similar_transform
from utils.functions import draw_landmarks, get_suffix

import numpy as np

if __name__ == "__main__":
  
  u_ref, w_shp_ref, w_exp_ref = load_origininal_bfm(
      '/home/innovplus/Dream/Projects/Data/300W_LP/Code/ModelGeneration/Model_Shape.mat',
      '/home/innovplus/Dream/Projects/Data/300W_LP/Code/ModelGeneration/Model_Exp.mat'
  )

  img, img_params, scale = load_image_and_its_params(
      '/home/innovplus/Dream/Projects/Data/300W_LP/AFW/AFW_815038_1_2.jpg',
      '/home/innovplus/Dream/Projects/Data/300W_LP/AFW/AFW_815038_1_2.mat',)

  R, offset, alpha_shp, alpha_exp = _parse_param(img_params)

  pts3d = scale * R @ (u_ref + w_shp_ref @ alpha_shp + w_exp_ref @ alpha_exp). \
                    reshape(3, -1, order='F') + offset

  draw_landmarks(img, [pts3d], show_flag=True, dense_flag=False, wfp=None)