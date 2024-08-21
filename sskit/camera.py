import torch
import numpy as np
import json
from sskit.utils import to_homogeneous, to_cartesian
from pathlib import Path

def world_to_undistorted(camera_matrix, pkt):
    return to_cartesian(torch.matmul(to_homogeneous(pkt), camera_matrix.mT))

def undistorted_to_ground(camera_matrix, pkt):
    hom = torch.inverse(camera_matrix[..., [0, 1, 3]])
    pkt = to_cartesian(torch.matmul(to_homogeneous(pkt), hom.mT))
    return torch.cat([pkt, torch.zeros_like(pkt[..., 0:1])], -1)

def distort(poly, pkt):
    rr = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr2 = polyval(poly, torch.arctan(rr))
    scale = rr2 / rr
    return scale * pkt

def undistort(poly, pkt):
    rr2 = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr = torch.tan(polyval(poly, rr2))
    scale = rr / rr2
    return scale * pkt


def polyval(poly, pkt):
    sa = poly[..., 0:1]
    for i in range(1, poly.shape[-1]):
        sa = pkt * sa + poly[..., i]
    return sa

def world_to_image(camera_matrix, distortion_poly, pkt):
    return distort(distortion_poly, world_to_undistorted(camera_matrix, pkt))

def image_to_ground(camera_matrix, undistortion_poly, pkt):
    return undistorted_to_ground(camera_matrix, undistort(undistortion_poly, pkt))

def load_camera(directory: Path, poly_dim=8):
    directory = Path(directory)
    camera_matrix = np.load(directory / "camera_matrix.npy")[:3]
    with open(directory / "lens.json") as fd:
        lens = json.load(fd)
    dist_poly = lens["dist_poly"]
    sensor_width = lens["sensor_width"]
    pixel_width = lens["pixel_width"]

    def d2a(dist_poly, x):
        return -sum(k * x ** i for i, k in enumerate(dist_poly)) / 180 * np.pi

    rr2 = np.linspace(0, 1.5, 200)
    rr = d2a(dist_poly, rr2 * sensor_width * pixel_width)
    msk = (0 <= rr) & (rr < 1.5)
    rr2 = rr2[msk]
    rr = rr[msk]
    poly = np.polyfit(rr, rr2, poly_dim)
    rev_poly = np.polyfit(rr2, rr, poly_dim)

    # import matplotlib.pyplot as plt
    # plt.plot(rr, rr2)
    # plt.plot(rr, np.polyval(poly, rr))
    # plt.show()

    t = torch.get_default_dtype()
    camera_matrix_t = torch.tensor(camera_matrix).to(t)
    poly_t = torch.tensor(poly).to(t)
    rev_poly_t = torch.tensor(rev_poly).to(t)
    return camera_matrix_t, poly_t, rev_poly_t

# def a2d(a, dist_poly):
#     a0 = -a * 180 / np.pi
#     x = (a0 - dist_poly[0]) / dist_poly[1]
#     for i in range(20):
#         x = (
#             a0 - sum(k * x ** i for i, k in enumerate(dist_poly) if i != 1)
#         ) / dist_poly[1]
#     return x

# def world_to_image_nopoly(d, x, y, z):
#     camera_matrix = np.load(d / "camera_matrix.npy")
#     with open(d / "lens.json") as fd:
#         lens = json.load(fd)
#     dist_poly = lens["dist_poly"]
#     sensor_width = lens["sensor_width"]
#     pixel_width = lens["pixel_width"]

#     x, y, z, _ = camera_matrix @ (x, y, z, 1)
#     cx = x / z
#     cy = y / z
#     print("cx, cy = ", cx, cy)

#     rr = np.sqrt(cx ** 2 + cy ** 2)
#     rr2 = a2d(np.arctan(rr), dist_poly) / pixel_width / sensor_width
#     print("rr, rr2 = ", rr, rr2)

#     cx *= rr2/rr
#     cy *= rr2/rr

#     height, width, _ = imshape(d / 'rgb.jpg')
#     principal = np.array([width / 2, height / 2])
#     return width * cx + principal[0], width * cy + principal[1]
