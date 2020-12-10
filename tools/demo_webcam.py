
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
#net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

cap = cv2.VideoCapture(2)
import time
t1 = time.time()
while True:
    ret, im_raw = cap.read()
    im = cv2.resize(im_raw, (160,90))
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    print(time.time() - t1)
    t1 = time.time()
    pred = palette[out]

    #pred = cv2.resize(pred, (640, 360))
    im_raw = cv2.resize(im_raw, (640, 360))

    cv2.imshow('frame', pred)
    #cv2.imshow('raw image', im_raw)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
