import torch
import numpy as np
import json
from sskit.utils import imshape, to_homogeneous, to_cartesian
from pathlib import Path

def world_to_undistorted(camera_matrix, pkt):
    return to_cartesian(torch.matmul(to_homogeneous(pkt), camera_matrix.mT))

def distort(poly, pkt):
    rr = (pkt ** 2).sum(-1, keepdim=True).sqrt()
    rr2 = polyval(poly, np.arctan(rr))
    # rr2 = polyval(poly, rr)
    print("rr, rr2 = ", rr.max(), rr2.max())
    scale = rr2 / rr
    return scale * pkt

def polyval(poly, pkt):
    sa = poly[..., 0:1]
    for i in range(1, poly.shape[-1]):
        sa = pkt * sa + poly[..., i]
    return sa

def world_to_image(camera_matrix, distortion_poly, pkt):
    return distort(distortion_poly, world_to_undistorted(camera_matrix, pkt))

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
    # rr = np.tan(d2a(dist_poly, rr2 * sensor_width * pixel_width))
    rr = d2a(dist_poly, rr2 * sensor_width * pixel_width)
    msk = (0 <= rr) & (rr < 1.5)
    rr2 = rr2[msk]
    rr = rr[msk]

    # poly = np.polyfit(np.arctan(rr), rr2, poly_dim)
    # (rr - np.polyval(poly, np.arctan(rr))).max()

    poly = np.polyfit(rr, rr2, poly_dim)
    # (rr2 - np.polyval(poly, rr)).max()

    # import matplotlib.pyplot as plt
    # plt.plot(rr, rr2)
    # plt.plot(rr, np.polyval(poly, rr))
    # plt.show()


    # import matplotlib.pyplot as plt
    # plt.plot(np.arctan(rr), rr2)
    # # plt.plot(np.arctan(rr), np.polyval(poly, rr))
    # plt.plot(np.arctan(rr), np.polyval(poly, np.arctan(rr)))
    # plt.show()


    t = torch.get_default_dtype()
    camera_matrix_t = torch.tensor(camera_matrix).to(t)
    poly_t = torch.tensor(poly).to(t)
    return camera_matrix_t, poly_t

d = Path("/home/hakan/data/SoccerCrowdV1/20240103_213931_node052_10173952_3348495_001_000/StudenternasCenterLeft/")
camera_matrix, dist_poly = load_camera(str(d))
pkt = torch.tensor([30.5681917364984, 31.9043639410341, 0.0356437899172306])

from sskit.utils import world_to_image_nopoly
world_to_image_nopoly(d, *pkt)

world_to_undistorted(camera_matrix, pkt)  # (0.21174133411913104, -0.210572232411525)
ipkt = world_to_image(camera_matrix, dist_poly, pkt)  # 0.1530545827352605, -0.15220951214582118

h, w, _ = imshape(d / 'rgb.jpg')
ipkt * w + torch.tensor([w/2, h/2])  # 2507.72959770331, 495.515473359975



    # for n in range(30):
    #     p = np.polyfit(rr, rr2, n)
    #     ad = np.abs(np.polyval(p, rr) - rr2) * w/2
    #     o = ad.max()
    #     print(n, o, sum(ad<1))

    # import matplotlib.pyplot as plt
    # plt.plot(rr, rr2)
    # plt.plot(rr, np.polyval(p, rr))
    # plt.show()




# cam = load_camera(d)

# class Cameras:
#     def __init__(self, camera_matrix, distortion_poly, sensor_scale):
#         self.camera_matrix = camera_matrix
#         self.distortion_poly = distortion_poly
#         self.sensor_scale = sensor_scale

#     def world_to_image(self, pkt):
#         return world_to_image(self.camera_matrix, self.distortion_poly, pkt)

#     def distort(self, pkt):
#         return distort(self.distortion_poly, pkt)
