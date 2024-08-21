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

