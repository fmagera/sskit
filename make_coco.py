from pathlib import Path
from sskit import imshape, world_to_image, load_camera
import json
from tqdm import tqdm

d = Path('/home/hakan/data/SoccerCrowdV1')
out = Path('/home/hakan/data/SpiideoScenes/SoccerCrowd/v1')

def mkcoco(part):
    imdir = out / part
    imdir.mkdir(parents=True)
    annotations = []
    images = []
    for iid, fn in enumerate(tqdm((d / f'{part}_v1.txt').open().readlines())):
        fn = d / fn.strip()
        ofn = imdir / f'{iid:06d}.jpg'
        ofn.hardlink_to(fn)

        c, h, w = imshape(fn)
        with open(fn.parent / "objects.json") as fd:
            objects = json.load(fd)
        camera_matrix, dist_poly, undist_poly = load_camera(fn.parent)

        images.append(dict(
            id=iid,
            file_name=ofn.name,
            width=w,
            height=h,
            camera_matrix=camera_matrix.tolist(),
            dist_poly=dist_poly.tolist(),
            undist_poly=undist_poly.tolist(),
        ))


        for obj in objects.values():
            if obj['class'] == 'human':
                for key in ["bounding_box_tighter", "bounding_box_tight"]:
                    if key in obj:
                        bbox = obj[key]
                        break
            else:
                continue
            u0, u1, v0, v1 = bbox
            box_w = u1 - u0
            box_h = v1 - v0
            px, py, pz = obj['keypoints']['pelvis']
            pu, pv = (world_to_image(camera_matrix, dist_poly, [px, py, pz]).numpy() * w + (w/2, h/2))
            pu2, pv2 = (world_to_image(camera_matrix, dist_poly, [px, py, 0]).numpy() * w + (w/2, h/2))

            annotations.append(dict(
                id=len(annotations),
                keypoints=[[pu, pv, 1], [pu2, pv2, 1]],
                bbox=[u0, v0, box_w, box_h],
                image_id=iid,
                category_id=1,
            ))



    (out / "annotations").mkdir(exist_ok=True)
    with open(out / "annotations" / f"{part}.json", "w") as fd:
        json.dump(dict(images=images, annotations=annotations, categories=[dict(id=1, name="person")]), fd)

mkcoco("train")
mkcoco("val")
mkcoco("test")
mkcoco("challange")
