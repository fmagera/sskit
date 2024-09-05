from pathlib import Path
import json
from random import shuffle
import sys

d = Path(sys.argv[1])

val_scenes = {"fcfef5cf-57ce-46ea-b437-b2523dc5ae43", "d097fb47-4284-46e5-aa1d-57a500cd9f63"}
tst_scenes = {"9e6244cc-b709-44d0-81b5-685e1c37f876", "21aef7e9-7f4a-4dcd-80f9-40d78325d2bc"}

with open("g2s.json") as fd:
    g2s = json.load(fd)

train, val, test = [], [], []
for fn in d.rglob('**/rgb.jpg'):
    with (fn.parent / 'scene_info.json').open() as fd:
        game = json.load(fd)["background_collected_from_game"]
    scene = g2s[game[0].split('/')[-1]]
    if scene in val_scenes:
        val.append(fn)
    elif scene in tst_scenes:
        test.append(fn)
    else:
        train.append(fn)

shuffle(val)
shuffle(test)
shuffle(train)

(d / "val_v1.txt").write_text('\n'.join(str(fn.relative_to(d)) for fn in val) + '\n')
(d / "test_v1.txt").write_text('\n'.join(str(fn.relative_to(d)) for fn in test) + '\n')
(d / "train_v1.txt").write_text('\n'.join(str(fn.relative_to(d)) for fn in train) + '\n')