import os
import cv2
import random
import numpy as np

images_folder = "C:/Users/Snow/Downloads/urban-waste.v5i.yolo26/urban-waste.v5i.yolo26/train/images"
labels_folder = "C:/Users/Snow/Downloads/urban-waste.v5i.yolo26/urban-waste.v5i.yolo26/train/labels"

out_img = "C:/Users/Snow/Downloads/urban-waste.v5i.yolo26/urban-waste.v5i.yolo26/train/aug_images"
out_lbl = "C:/Users/Snow/Downloads/urban-waste.v5i.yolo26/urban-waste.v5i.yolo26/train/aug_labels"

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_lbl, exist_ok=True)

#label-based strength
AUGMENTATION_MAP = {
    4: ["flip_h", "flip_v", "rot90", "scale", "translate"],
    0: ["flip_h", "rot90", "scale"],
    1: ["flip_h", "rot90"],
    3: ["flip_h", "scale", "translate"],
}


def flip_h(coords):
    return [1-x if i % 2 == 0 else x for i, x in enumerate(coords)]

def flip_v(coords):
    return [1-x if i % 2 == 1 else x for i, x in enumerate(coords)]

def rot90(coords):
    new = []
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]
        new += [y, 1-x]
    return new

def scale_coords(coords, scale=1.2):
    new = []
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]

        x = (x - 0.5) * scale + 0.5
        y = (y - 0.5) * scale + 0.5

        new += [x, y]
    return new

def translate_coords(coords, dx=0.1, dy=0.1):
    new = []
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]

        x = x + dx
        y = y + dy

        new += [x, y]
    return new

def clip_coords(coords):
    return [min(1, max(0, x)) for x in coords]

#main loop

for file in os.listdir(labels_folder):
    if not file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_folder, file)
    image_path = os.path.join(images_folder, file.replace(".txt", ".jpg"))

    if not os.path.exists(image_path):
        continue

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    labels = [int(l.split()[0]) for l in lines]

    selected_transforms = set()
    for label in labels:
        if label in AUGMENTATION_MAP:
            selected_transforms.update(AUGMENTATION_MAP[label])

    if not selected_transforms:
        continue

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    base_name = file.replace(".txt", "")

    for idx, aug in enumerate(selected_transforms, start=1):

        img_aug = img.copy()
        new_lines = []

        #random params
        scale_factor = random.uniform(1.1, 1.4)
        dx = random.uniform(-0.1, 0.1)
        dy = random.uniform(-0.1, 0.1)

        if aug == "flip_h":
            img_aug = cv2.flip(img_aug, 1)

        elif aug == "flip_v":
            img_aug = cv2.flip(img_aug, 0)

        elif aug == "rot90":
            img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_CLOCKWISE)

        elif aug == "scale":
            nh, nw = int(h * scale_factor), int(w * scale_factor)
            img_aug = cv2.resize(img_aug, (nw, nh))

            #center crop back
            start_x = (nw - w) // 2
            start_y = (nh - h) // 2
            img_aug = img_aug[start_y:start_y+h, start_x:start_x+w]

        elif aug == "translate":
            M = [[1, 0, dx*w], [0, 1, dy*h]]
            img_aug = cv2.warpAffine(img_aug,
                                     np.array(M, dtype=float),
                                     (w, h))

        for line in lines:
            parts = line.split()
            label = parts[0]
            coords = list(map(float, parts[1:]))

            if aug == "flip_h":
                coords = flip_h(coords)

            elif aug == "flip_v":
                coords = flip_v(coords)

            elif aug == "rot90":
                coords = rot90(coords)

            elif aug == "scale":
                coords = scale_coords(coords, scale_factor)

            elif aug == "translate":
                coords = translate_coords(coords, dx, dy)

            coords = clip_coords(coords)

            new_lines.append(label + " " + " ".join(map(str, coords)))

        suffix = f"_{idx:02d}"
        img_name = base_name + suffix + ".jpg"
        lbl_name = base_name + suffix + ".txt"

        cv2.imwrite(os.path.join(out_img, img_name), img_aug)

        with open(os.path.join(out_lbl, lbl_name), "w") as f:
            f.write("\n".join(new_lines))

print("Done!")