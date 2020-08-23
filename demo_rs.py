import sys
sys.path.append('core')

import argparse
import os
import cv2
import yaml
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config_folder as cf
from data_loaders.Chairs import Chairs
from data_loaders.kitti import KITTI
from data_loaders.sintel import Sintel
from model import MaskFlownet, MaskFlownet_S, Upsample, EpeLossWithMask

import pyrealsense2 as rs
from imutils.video import FPS
from utils import flow_viz


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def centralize(img1, img2):
    rgb_mean = torch.cat((img1, img2), 2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
    return img1 - rgb_mean, img2-rgb_mean, rgb_mean

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)


def demo(args):
    resize = (int(args.resize.split(',')[0]), int(args.resize.split(',')[1])) if args.resize else None
    num_workers = 2

    with open(os.path.join('config_folder', args.dataset_cfg)) as f:
        config = cf.Reader(yaml.load(f))
    with open(os.path.join('config_folder', args.config)) as f:
        config_model = cf.Reader(yaml.load(f))

    net = eval(config_model.value['network']['class'])(config)
    checkpoint = torch.load(os.path.join('weights', args.checkpoint))

    net.load_state_dict(checkpoint)
    net = net.to(device)

    fps = None

    with torch.no_grad():
        fps = FPS().start()

        try:
            image1 = None
            image2 = None
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                im = np.asanyarray(color_frame.get_data())[:,:,::-1].astype(np.uint8)
                img = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0)


                if image2 is None:
                    image1 = img
                else:
                    image1 = image2

                image2 = img

                im0, im1, _ = centralize(image1, image2)
                shape = im0.shape
                pad_h = (64 - shape[2] % 64) % 64
                pad_w = (64 - shape[3] % 64) % 64
                if pad_h != 0 or pad_w != 0:
                    im0 = F.interpolate(im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
                    im1 = F.interpolate(im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')

                im0 = im0.to(device)
                im1 = im1.to(device)

                pred, flows, warpeds = net(im0, im1)

                up_flow = Upsample(pred[-1], 4)
                up_occ_mask = Upsample(flows[0], 4)

                if pad_h != 0 or pad_w != 0:
                    up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
                              torch.tensor([shape[d] / up_flow.shape[d] for d in (2, 3)], device=device).view(1, 2, 1,
                                                                                                              1)
                    up_occ_mask = F.interpolate(up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')

                fps.update()
                fps.stop()
                print(f"{fps.fps():.2f}")
                viz(image1, up_flow)




                # check for escape key
                key = cv2.waitKey(1)
                if key == 27 or key == 1048603:
                    break
        finally:

            # Stop streaming
            pipeline.stop()
            # release the pointer
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str, nargs='?', default=None)
    parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='model checkpoint to load')
    parser.add_argument('-b', '--batch', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('-f', '--root_folder', type=str, default=None,
                        help='Root folder of KITTI')
    parser.add_argument('--resize', type=str, default='')
    args = parser.parse_args()


    demo(args)
