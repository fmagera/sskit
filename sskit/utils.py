from torchvision.io import read_image
from torchvision.utils import save_image
import torch
import PIL.Image

def imread(fn):
    return read_image(fn).to(torch.get_default_dtype()) / 255

def imwrite(img, fn, rescale=False):
    if rescale:
        img = img - img.min()
        img /= img.max()
    save_image(img, fn)

def immark(im, pos, color=(1,0,0), r=3):
    _, h, w = im.shape
    pos = torch.atleast_2d(pos)
    assert pos.shape[-1] == 2
    for u,v in pos.reshape(-1, 2):
        u, v = int(torch.round(u)), int(torch.round(v))
        if 0 <= u < w and 0 <= v < h:
            im[:, max(v-r, 0):min(v+r, h-1), max(u-r, 0):min(u+r,w-1)] = torch.tensor(color)[:, None, None]
    return im

def imshape(fn):
    im = PIL.Image.open(fn)
    w, h = im.size
    if im.mode != 'RGB':
        raise NotImplementedError
    return (h, w, 3)

def to_homogeneous(pkt):
    return torch.cat([pkt, torch.ones_like(pkt[..., 0:1])], -1)

def to_cartesian(pkt):
    return pkt[..., :-1] / pkt[..., -1:]



# fn = d / 'rgb.jpg'

# pkt = torch.tensor([5681917364984, 31.9043639410341, 0.0356437899172306])
# to_cartesian(pkt[None,None]).shape

# to_homogeneous(pkt[None].repeat(2,1)[None][:,:,:2])

# p = to_homogeneous(pkt[None].repeat(2, 1))
# cam = torch.tensor(camera_matrix).to(torch.float32)


# torch.matmul(p, cam.mT)

# torch.matmul(cam, torch.transpose(p[None], -2, -1)).mT


# [pkt, pkt])

# from pathlib import Path
# d = Path("/home/hakan/data/SoccerCrowdV1/20240103_213931_node052_10173952_3348495_001_000/StudenternasCenterLeft/")
# im = imread(d / "rgb.jpg")
# imwrite(im[0,:,:], 't.jpg')

import numpy as np
import json


def a2d(a, dist_poly):
    a0 = -a * 180 / np.pi
    x = (a0 - dist_poly[0]) / dist_poly[1]
    for i in range(20):
        x = (
            a0 - sum(k * x ** i for i, k in enumerate(dist_poly) if i != 1)
        ) / dist_poly[1]
    return x

def world_to_image_nopoly(d, x, y, z):
    camera_matrix = np.load(d / "camera_matrix.npy")
    with open(d / "lens.json") as fd:
        lens = json.load(fd)
    dist_poly = lens["dist_poly"]
    sensor_width = lens["sensor_width"]
    pixel_width = lens["pixel_width"]

    x, y, z, _ = camera_matrix @ (x, y, z, 1)
    cx = x / z
    cy = y / z
    print("cx, cy = ", cx, cy)

    rr = np.sqrt(cx ** 2 + cy ** 2)
    # arg = np.arctan2(cy, cx)
    rr2 = a2d(np.arctan(rr), dist_poly) / pixel_width / sensor_width
    # cx =  rr2 * np.cos(arg)
    # cy =  rr2 * np.sin(arg)
    print("rr, rr2 = ", rr, rr2)

    cx *= rr2/rr
    cy *= rr2/rr

    height, width, _ = imshape(d / 'rgb.jpg')
    principal = np.array([width / 2, height / 2])
    return width * cx + principal[0], width * cy + principal[1]

# # ox, oy =
# x, y, z = 30.5681917364984, 31.9043639410341, 0.0356437899172306
# world_to_image(30.5681917364984, 31.9043639410341, 0.0356437899172306)  # 2507.72959770331, 495.515473359975
# imwrite(immark(im.clone(), (ox, oy)), "t.png")
