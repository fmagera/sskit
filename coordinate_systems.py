import torch
from sskit import load_camera, imread, world_to_image, Draw, image_to_ground, undistort_image, imwrite, project_on_ground, normalize, unnormalize, undistort, undistorted_to_ground
from pathlib import Path

d = Path("example")
camera_matrix, dist_poly, undist_poly = load_camera(d)
img = imread(d / "rgb.jpg")
_, h, w = img.shape
pkt = unnormalize(world_to_image(camera_matrix, dist_poly, [0.0, 0.0, 0.0]), img.shape)
off = torch.tensor([40, -40])
Draw(img).circle(pkt, 10, 'red').text([40, 40], '(0, 0)').text([w-400, h-120], f'({w}, {h})').text(pkt+off, f'({pkt[0]:.1f}, {pkt[1]:.1f})').save("docs/camera.jpg")

u1, v1 = normalize([0, 0], img.shape)
u2, v2 = normalize([w, h], img.shape)
normalized_pkt = normalize(pkt, img.shape)
Draw(img).circle(pkt, 10, 'red').text([40, 40], f'({u1:.1f}, {v1:.2f})').text([w-400, h-120], f'({u2:.1f}, {v2:.2f})').text(pkt+off, f'({normalized_pkt[0]:.1f}, {normalized_pkt[1]:.1f})').save("docs/normalized.jpg")

zoom = 0.5
undistorted_img = undistort_image(dist_poly, img[None], zoom)[0]
undistorted_pkt = undistort(undist_poly, normalized_pkt)
upkt = unnormalize(undistorted_pkt * zoom, img.shape)
Draw(undistorted_img).circle(upkt, 10, 'red').text(upkt+off, f'({undistorted_pkt[0]:.1f}, {undistorted_pkt[1]:.1f})').save("docs/undistorted.jpg")

ground_img = project_on_ground(camera_matrix, dist_poly, img[None])[0]
ground_pkt = undistorted_to_ground(camera_matrix, undistorted_pkt)
gpkt = torch.tensor([700/2, 1200/2])
Draw(ground_img).circle(gpkt, 10, 'red').text(gpkt+off, f'({ground_pkt[0]:.1f}, {ground_pkt[1]:.1f})').save("docs/ground.jpg")
