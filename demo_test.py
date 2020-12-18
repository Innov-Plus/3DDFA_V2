from training.prepare_data import *
from training.export_bfm import *
from utils.tddfa_util import _parse_param, similar_transform
from utils.functions import draw_landmarks, get_suffix, crop_img

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

  customBFMModel = CustomBFMModel()

  image = Image3DDFA(
      '/home/innovplus/Dream/Projects/Data/300W_LP/AFW/AFW_815038_1_2.jpg',
      '/home/innovplus/Dream/Projects/Data/300W_LP/AFW/AFW_815038_1_2.mat',)

  image.calculate68Points(customBFMModel)

  box = image.preprocess(customBFMModel)

  cropped_img = crop_img(image.img, box)

  draw_landmarks(cropped_img, [image.pts3d], show_flag=True, dense_flag=False, wfp=None)