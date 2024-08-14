from torchvision.io import read_image
from torchvision.utils import save_image
import torch

def imread(fn):
    return read_image(fn).to(torch.get_default_dtype()) / 255

def imwrite(img, fn, rescale=False):
    if rescale:
        img = img - img.min()
        img /= img.max()
    save_image(img, fn)

def immark(im, pos, color=(1,0,0), r=3):
    u, v = pos
    u, v = round(u), round(v)
    _, h, w = im.shape
    if 0 <= u < w and 0 <= v < h:
        im[:, max(v-r, 0):min(v+r, h-1), max(u-r, 0):min(u+r,w-1)] = torch.tensor(color)[:, None, None]
    return im

from pathlib import Path
d = Path("/home/hakan/data/SoccerCrowdV1/20240103_213931_node052_10173952_3348495_001_000/StudenternasCenterLeft/")
im = imread(d / "rgb.jpg")
imwrite(im[0,:,:], 't.jpg')

import numpy as np
import json

camera_matrix = np.load(d / "camera_matrix.npy")

def world_to_image(x, y, z):
    with open(d / "lens.json") as fd:
        lens = json.load(fd)
    dist_poly = lens["dist_poly"]
    sensor_width = lens["sensor_width"]
    pixel_width = lens["pixel_width"]

    _, height, width = im.shape
    image_scale = sensor_width / width
    principal = np.array([width / 2, height / 2])
    x, y, z, _ = camera_matrix @ (x, y, z, 1)
    cx = x / z
    cy = y / z
    rr = np.sqrt(cx ** 2 + cy ** 2)
    arg = np.arctan2(cy, cx)
    rr = a2d(np.arctan(rr), dist_poly) / pixel_width / sensor_width
    cx = width * rr * np.cos(arg)
    cy = width * rr * np.sin(arg)
    return cx + principal[0], cy + principal[1]

def a2d(a, dist_poly):
    a0 = -a * 180 / np.pi
    x = (a0 - dist_poly[0]) / dist_poly[1]
    for i in range(20):
        x = (
            a0 - sum(k * x ** i for i, k in enumerate(dist_poly) if i != 1)
        ) / dist_poly[1]
    return x

# ox, oy =
world_to_image(30.5681917364984, 31.9043639410341, 0.0356437899172306)  # 2507.72959770331, 495.515473359975
imwrite(immark(im.clone(), (ox, oy)), "t.png")
