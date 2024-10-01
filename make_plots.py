import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from soccerpitch import SoccerPitch
from sskit import load_camera, imread, world_to_image, Draw, image_to_ground

soccer_scenes_dir = "/home/fmg/data/SoccerSceneV1"

for node_dir in os.listdir(soccer_scenes_dir):

    full_node_dir = Path(f"{soccer_scenes_dir}/{node_dir}")
    side_results = {}
    sides_dir = os.listdir(full_node_dir)
    ordered_sides = [[d for d in sides_dir if "Left" in d][0], [d for d in sides_dir if "Right" in d][0]]
    for side in ordered_sides:
        d = Path(f"{full_node_dir}/{side}")
        camera_matrix, dist_poly, undist_poly = load_camera(d)

        img = imread(d / "rgb.jpg")
        _, h, w = img.shape

        with open(f"{d}/scene_info.json", "r") as f:
            scene_info = json.load(f)

        pitch_corner_position = scene_info["roi"][0][0]
        sp = SoccerPitch(
            abs(pitch_corner_position[1]) * 2,
            abs(pitch_corner_position[0]) * 2
        )
        points = sp.sample_field_points()

        drw = Draw(img)
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0

        projected_lines = []

        for key, polyline in points.items():
            if "post" in key or "crossbar" in key:
                continue
            polyline = [np.array([point[1], point[0], point[2]]) for point in polyline]
            torch_poly = torch.from_numpy(np.array(polyline))

            projected = w * world_to_image(camera_matrix, dist_poly, torch_poly.float()) + torch.tensor([w / 2, h / 2])
            mask = torch.logical_and(projected[:, 0] < w, projected[:, 1] < h)
            mask = torch.logical_and(mask, projected[:, 0] >= 0)
            mask = torch.logical_and(mask, projected[:, 1] >= 0)
            selected = projected[mask]

            if selected.size(0) > 0:
                projected_lines.append(selected)
                drw.circle(selected, 3, (0, 0, 255))
                drw.line(selected, (0, 0, 255), 2)
                if torch.amin(selected[:, 0], dim=0).item() < x_min:
                    x_min, _ = torch.min(selected[:, 0], dim=0)
                if torch.amin(selected[:, 1], dim=0).item() < y_min:
                    y_min, _ = torch.min(selected[:, 1], dim=0)

                if torch.amax(selected[:, 0], dim=0).item() > x_max:
                    x_max, _ = torch.max(selected[:, 0], dim=0)
                if torch.amax(selected[:, 1], dim=0).item() > y_max:
                    y_max, _ = torch.max(selected[:, 1], dim=0)

        xrange = torch.arange(x_min, x_max)
        yrange = torch.arange(y_min, y_max)

        coordinates = torch.meshgrid(xrange, yrange, indexing="ij")
        coordinates = torch.stack(coordinates, -1)
        coordinates = coordinates.view(-1, 2)

        ground_coordinates = image_to_ground(camera_matrix, undist_poly,
                                             (coordinates - torch.tensor([w / 2, h / 2])) / w)

        ground_coordinates = ground_coordinates.reshape(xrange.size(0), yrange.size(0), 3)


        def distance(point1, point2):
            return torch.sqrt(torch.sum(torch.square(point1 - point2)))


        values = torch.zeros([ground_coordinates.size(0) // 10, ground_coordinates.size(1) // 10])

        for i in tqdm(range(0, ground_coordinates.size(0) - 10, 10)):
            for j in range(0, ground_coordinates.size(1) - 10, 10):
                neighbours = []
                if ground_coordinates.size(0) - 1 > i > 0:
                    neighbours.append(ground_coordinates[i - 1, j])
                    neighbours.append(ground_coordinates[i + 1, j])
                    if ground_coordinates.size(1) - 1 > j > 0:
                        neighbours.append(ground_coordinates[i - 1, j - 1])
                        neighbours.append(ground_coordinates[i + 1, j - 1])
                        neighbours.append(ground_coordinates[i + 1, j + 1])
                        neighbours.append(ground_coordinates[i - 1, j + 1])
                        neighbours.append(ground_coordinates[i, j - 1])
                        neighbours.append(ground_coordinates[i, j + 1])
                    elif j > 0:
                        neighbours.append(ground_coordinates[i, j - 1])
                        neighbours.append(ground_coordinates[i + 1, j - 1])
                        neighbours.append(ground_coordinates[i - 1, j - 1])
                    elif j == 0:
                        neighbours.append(ground_coordinates[i, j + 1])
                        neighbours.append(ground_coordinates[i + 1, j + 1])
                        neighbours.append(ground_coordinates[i - 1, j + 1])
                elif i == 0:
                    neighbours.append(ground_coordinates[i + 1, j])
                    if ground_coordinates.size(1) - 1 > j > 0:
                        neighbours.append(ground_coordinates[i + 1, j - 1])
                        neighbours.append(ground_coordinates[i + 1, j + 1])
                        neighbours.append(ground_coordinates[i, j - 1])
                        neighbours.append(ground_coordinates[i, j + 1])
                    elif j == ground_coordinates.size(1) - 1:
                        neighbours.append(ground_coordinates[i, j - 1])
                        neighbours.append(ground_coordinates[i + 1, j - 1])
                    elif j == 0:
                        neighbours.append(ground_coordinates[i, j + 1])
                        neighbours.append(ground_coordinates[i + 1, j + 1])
                elif i == ground_coordinates.size(0) - 1:
                    neighbours.append(ground_coordinates[i - 1, j])
                    if ground_coordinates.size(1) - 1 > j > 0:
                        neighbours.append(ground_coordinates[i - 1, j - 1])
                        neighbours.append(ground_coordinates[i - 1, j + 1])
                        neighbours.append(ground_coordinates[i, j - 1])
                        neighbours.append(ground_coordinates[i, j + 1])
                    elif j == ground_coordinates.size(1) - 1:
                        neighbours.append(ground_coordinates[i, j - 1])
                        neighbours.append(ground_coordinates[i - 1, j - 1])
                    elif j == 0:
                        neighbours.append(ground_coordinates[i, j + 1])
                        neighbours.append(ground_coordinates[i - 1, j + 1])
                distances = torch.tensor([distance(neighbour, ground_coordinates[i, j]) for neighbour in neighbours])
                values[i // 10, j // 10] = torch.mean(distances)

        xrange = torch.arange(x_min, x_max - 9, 10)
        yrange = torch.arange(y_min, y_max - 9, 10)
        side_results[side] = (values, projected_lines, xrange, yrange)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # gr2red = cm.get_cmap('RdYlGn_r')
    gr2red = colormaps['RdYlGn_r']
    new_gr2red = ListedColormap(gr2red(np.linspace(0.55, 1, 128)))

    for i, (values, projected_lines, xrange, yrange) in enumerate(side_results.values()):

        axs[i].set(xlim=(xrange[0], xrange[-1]), ylim=(yrange[0], yrange[-1]), xticks=[], yticks=[])
        levels = np.linspace(0, 0.5, 20)

        cs = axs[i].contourf(xrange, yrange, values.transpose(0, 1), levels, extend="both",
                             cmap=gr2red)  # cm.RdYlGn_r) #cm.YlOrBr)

        for line in projected_lines:
            X = [l[0] for l in line]
            Y = [l[1] for l in line]
            axs[i].plot(X, Y, linestyle='solid', color='blue')
        axs[i].invert_yaxis()

    fig.colorbar(cs, ax=axs[-1], ticks=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

    fig.tight_layout()
    fig.savefig(f"./plots/{node_dir}.png")
    plt.close()
