from pathlib import Path
from sskit import load_camera, imread, imwrite, world_to_image, immark
import json
import torch

d = Path("/home/hakan/data/SoccerCrowdV1/20240103_213931_node052_10173952_3348495_001_000/StudenternasCenterLeft/")
camera_matrix, dist_poly = load_camera(str(d), 8)

img = imread(d / "rgb.jpg")
objects = json.loads((d / "objects.json").read_bytes())
pkt = torch.tensor([obj['keypoints'].get('left_foot') for obj in objects.values() if obj['class'] == 'human'])
pkt_img = torch.tensor([obj['keypoints'].get('left_foot_img') for obj in objects.values() if obj['class'] == 'human'])

npkt = world_to_image(camera_matrix[None], dist_poly[None], pkt)
_, h, w = img.shape
ipkt = npkt * w + torch.tensor([w/2, h/2])

print(((ipkt - pkt_img)**2).sum(2).sqrt().max())
immark(img, ipkt)
imwrite(img, "t.png")