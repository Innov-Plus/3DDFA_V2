# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import time
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn

import models
from utils.io import _load
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from utils.tddfa_util import (
    load_model,
    ToTensorGjz, NormalizeGjz,
    recon_dense, recon_sparse
)

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        torch.set_grad_enabled(False)

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),
            widen_factor=kvs.get('widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        model = load_model(model, kvs.get('checkpoint_fp'))

        if self.gpu_mode:
            cudnn.benchmark = True
            model = model.cuda(device=self.gpu_id)

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def __call__(self, img_ori, objs, **kvs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: the list of box or landmarks
        :param kvs: options
        :return: param list and roi_box list
        """
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = crop_img(img_ori, roi_box)
            print(img_ori.shape)


            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)            

            inp = self.transform(img).unsqueeze(0)

            with open("inp_py.txt", "w") as f:
                for i in range(inp.shape[1]):
                    for j in range(inp.shape[2]):
                        for k in range(inp.shape[3]):
                            f.write("{:.6f}".format(float(inp[0, i, j, k])))
                            f.write("\n")
                            # inp[0, i, j, k] = 0.0

            if self.gpu_mode:
                inp = inp.cuda(device=self.gpu_id)

            if kvs.get('timer_flag', False):
                end = time.time()
                param = self.model(inp)
                elapse = f'Inference: {(time.time() - end) * 1000:.1f}ms'
                print(elapse)
            else:
                param = self.model(inp)

            print('input', inp)    
            print('output', param)

            # with open("out_cpp.bin", "rb") as f:
            #     c = f.read()
            #     for i in range(62):
            #         print(float(c[i*4:i*4+4]))

            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            # param = [0.37304688, -0.16223145, 0.54638672, 0.11541748, 0.3269043, -1.8251953, -0.32250977, -1.9726562, -0.49707031, 0.24597168, 0.36010742, -0, 1.0419922, 0.34863281, -0.51513672, -0.023498535, 0.44042969, -0.26513672, 0.44970703, -0.14526367, 0.69970703, 0.011856079, 0.024917603, -0.10998535, 0.22045898, 0.13867188, 0.17626953, 0.051269531, -0.13916016, 0.1071167, 0.19494629, 0.032714844, 0.07244873, -0.11804199, -0.15197754, 0.25317383, -0.24682617, 0.12792969, 0.10931396, -0.13061523, -0.11743164, -0.12084961, -0.0094146729, -0.049194336, 0.088378906, -0.22314453, 0.1854248, 0.0088348389, 0.21130371, -0.19946289, -0.052490234, 0.105896, 0.78857422, -0.79248047, 0.42871094, -0.38549805, -0.21655273, 0.61865234, 0.53417969, 0.51757812, 0.22998047, 0.58984375]
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            if dense_flag:
                pts3d = recon_dense(param, roi_box, size)
            else:
                pts3d = recon_sparse(param, roi_box, size)  # 68 pts
                print(pts3d)

            ver_lst.append(pts3d)

        return ver_lst
