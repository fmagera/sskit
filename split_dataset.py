from pathlib import Path
import json
from random import shuffle
import sys
from collections import defaultdict

d = Path(sys.argv[1])

challange_scenes = {"fcfef5cf-57ce-46ea-b437-b2523dc5ae43", "d097fb47-4284-46e5-aa1d-57a500cd9f63"}
tst_scenes = {"9e6244cc-b709-44d0-81b5-685e1c37f876", "21aef7e9-7f4a-4dcd-80f9-40d78325d2bc"}
bad = {"allsvenskan/46837f26-01a6-45e7-a027-935dac58e211"}

with open("g2s.json") as fd:
    g2s = json.load(fd)

counts = defaultdict(int)
train, val, challange, test = [], [], [], []
for fn in d.rglob('**/rgb.jpg'):
    with (fn.parent / 'scene_info.json').open() as fd:
        game = json.load(fd)["background_collected_from_game"]
    counts[game[0][-1]] += 1
    scene = g2s[game[0].split('/')[-1]]
    if game[0] in bad:
        pass
    elif scene in challange_scenes or game[0][-1] in '0':
        challange.append(fn)
    elif scene in tst_scenes or game[0][-1] in '1':
        test.append(fn)
    elif game[0][-1] in '23':
        val.append(fn)
    else:
        train.append(fn)

shuffle(train)
shuffle(val)
shuffle(test)
shuffle(challange)
print(counts)
print(f'{len(train)} train, {len(val)} val, {len(test)} test, {len(challange)} challange')

(d / "train_v1.txt").write_text('\n'.join('./' + str(fn.relative_to(d)) for fn in train) + '\n')
(d / "val_v1.txt").write_text('\n'.join('./' + str(fn.relative_to(d)) for fn in val) + '\n')
(d / "test_v1.txt").write_text('\n'.join('./' + str(fn.relative_to(d)) for fn in test) + '\n')
(d / "challange_v1.txt").write_text('\n'.join('./' + str(fn.relative_to(d)) for fn in challange) + '\n')
