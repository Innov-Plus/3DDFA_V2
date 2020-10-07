# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def main(args):
    # Init TDDFA
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)

    # Init FaceBoxes
    face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    img.tofile("img.bin")

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    # boxes = [[228, 103, 467, 391, 1.0]]
    # print(boxes)
    n = len(boxes)
    print(f'Detect {n} faces')
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)

    param_lst, roi_box_lst = tddfa(img, boxes)

    # print(param_lst)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':
        # if `with_bf_flag` is False, the background is black
        depth(img, ver_lst, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')

    args = parser.parse_args()
    main(args)
