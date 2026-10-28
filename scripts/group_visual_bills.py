import cv2
import numpy as np
from pathlib import Path

# Compare all 35 images by downsampled color/structure similarity
uploads = sorted(Path("uploads").glob("*.jpeg"))
images = {}
for p in uploads:
    im = cv2.imread(str(p))
    if im is not None:
        images[p.name] = cv2.resize(im, (200, 200))

# Group identical/near-identical bills
groups = []
visited = set()
names = list(images.keys())

for i, n1 in enumerate(names):
    if n1 in visited:
        continue
    group = [n1]
    visited.add(n1)
    im1 = images[n1]
    for j in range(i + 1, len(names)):
        n2 = names[j]
        if n2 in visited:
            continue
        im2 = images[n2]
        # Check all 4 rotations
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rot_im = im2 if rot is None else cv2.rotate(im2, rot)
            diff = np.mean(np.abs(im1.astype(float) - rot_im.astype(float)))
            if diff < 30.0:  # Visual threshold for same photo/bill
                group.append(n2)
                visited.add(n2)
                break
    groups.append(group)

print(f"Total visual bill groups among 35 uploads: {len(groups)}")
for idx, g in enumerate(groups, 1):
    print(f"Group {idx} ({len(g)} files): {g[0]}")
