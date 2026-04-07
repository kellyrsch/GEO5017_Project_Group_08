import os
import cv2
import random
import numpy as np
import skimage as ski

images_folder = "data/urban-waste.v5i.yolo26/train/images"
labels_folder = "data/urban-waste.v5i.yolo26/train/labels"

out_img = "data/urban-waste.v5i.yolo26/train/aug_copypaste_images"
out_lbl = "data/urban-waste.v5i.yolo26/train/aug_copypaste_labels"

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_lbl, exist_ok=True)

# Next part is a script start/structure written by ChatGPT as framework to maybe continue working with, i'm a bit lost T-T

# =========================
# ⚙️ SETTINGS
# =========================

MINORITY_CLASSES = [0, 1, 3, 4]  # skip garbage bags (2)
MAX_OBJECTS_TO_PASTE = 2  # per image


# =========================
# 🧩 STEP 1: LOAD DATASET INDEX
# =========================

def load_dataset():
    """
    Returns:
        dataset = [
            {
                "image_path": ...,
                "label_path": ...
            },
            ...
        ]
    """
    dataset = []

    for file in os.listdir(labels_folder):
        if not file.endswith(".txt"):
            continue

        img_path = os.path.join(images_folder, file.replace(".txt", ".jpg"))
        lbl_path = os.path.join(labels_folder, file)

        if os.path.exists(img_path):
            dataset.append({
                "image_path": img_path,
                "label_path": lbl_path
            })

    return dataset


# =========================
# 🧩 STEP 2: BUILD OBJECT POOL
# =========================

def build_object_pool(dataset):
    """
    Returns:
        pool[class_id] = [
            {
                "image_path": ...,
                "polygon": [x1, y1, x2, y2, ...],
                "class": int
            }
        ]
    """
    pool = {c: [] for c in MINORITY_CLASSES}

    for item in dataset:
        with open(item["label_path"]) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            if cls in pool:
                pool[cls].append({
                    "image_path": item["image_path"],
                    "polygon": coords,
                    "class": cls
                })

    return pool


# =========================
# 🧩 STEP 3: POLYGON → MASK
# =========================

def polygon_to_mask(image_shape, polygon):
    """
    Convert normalized polygon → binary mask
    """
    h, w = image_shape[:2]

    pts = []
    for i in range(0, len(polygon), 2):
        x = int(polygon[i] * w)
        y = int(polygon[i + 1] * h)
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask, pts


# =========================
# 🧩 STEP 4: EXTRACT OBJECT
# =========================

def extract_object(image, mask):
    """
    Extract object using mask
    """
    obj = cv2.bitwise_and(image, image, mask=mask)

    # 👉 OPTIONAL (recommended):
    # Crop to bounding box of mask for efficiency
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    obj_crop = obj[y_min:y_max, x_min:x_max]
    mask_crop = mask[y_min:y_max, x_min:x_max]

    return obj_crop, mask_crop, (x_min, y_min)


# =========================
# 🧩 STEP 5: TRANSFORM OBJECT
# =========================

def transform_object(obj, mask):
    """
    Apply random transforms (scale, flip, etc.)

    👉 You should:
    - scale object
    - maybe flip
    - maybe rotate (small angles)

    IMPORTANT:
    You must apply SAME transform to mask
    """

    # Example: random horizontal flip
    if random.random() < 0.5:
        obj = cv2.flip(obj, 1)
        mask = cv2.flip(mask, 1)

    # 👉 TODO: implement scaling
    # hint:
    # scale = random.uniform(0.5, 1.2)
    # cv2.resize()

    # 👉 TODO: optional small rotation
    # hint:
    # cv2.getRotationMatrix2D()

    return obj, mask


# =========================
# 🧩 STEP 6: CHOOSE PASTE LOCATION
# =========================

def get_paste_position(target_shape, obj_shape):
    """
    Returns top-left position where object will be pasted
    """

    H, W = target_shape[:2]
    h, w = obj_shape[:2]

    # 👉 IMPORTANT:
    # Keep object inside bounds
    x = random.randint(0, max(1, W - w))

    # 👉 Better realism:
    # bias towards lower half of image
    y = random.randint(H // 2, max(H // 2 + 1, H - h))

    return x, y


# =========================
# 🧩 STEP 7: PASTE OBJECT
# =========================

def paste_object(target, obj, mask, x, y):
    """
    Paste object into target using mask
    """

    h, w = obj.shape[:2]

    roi = target[y:y + h, x:x + w]

    # paste only where mask is non-zero
    roi[mask > 0] = obj[mask > 0]

    target[y:y + h, x:x + w] = roi

    return target


# =========================
# 🧩 STEP 8: UPDATE POLYGON
# =========================

def update_polygon(polygon, offset, target_shape):
    """
    Convert original polygon → new location

    👉 Steps:
    1. Convert normalized → pixel coords
    2. Add offset
    3. Convert back to normalized
    """

    H, W = target_shape[:2]
    dx, dy = offset

    new_coords = []

    for i in range(0, len(polygon), 2):
        x = polygon[i] * W
        y = polygon[i + 1] * H

        # 👉 TODO:
        # This is WRONG if you cropped object earlier
        # You must account for crop offset!

        x_new = (x + dx) / W
        y_new = (y + dy) / H

        new_coords += [x_new, y_new]

    return new_coords


# =========================
# 🧩 STEP 9: LOAD LABELS
# =========================

def load_labels(label_path):
    with open(label_path) as f:
        return [l.strip() for l in f if l.strip()]


# =========================
# 🧩 STEP 10: MAIN LOOP
# =========================

def run_copy_paste():
    dataset = load_dataset()
    pool = build_object_pool(dataset)

    for idx, item in enumerate(dataset):

        target_img = cv2.imread(item["image_path"])
        target_labels = load_labels(item["label_path"])

        new_labels = target_labels.copy()

        for _ in range(MAX_OBJECTS_TO_PASTE):

            cls = random.choice(MINORITY_CLASSES)
            if not pool[cls]:
                continue

            obj_data = random.choice(pool[cls])

            # load source image
            src_img = cv2.imread(obj_data["image_path"])

            mask, pts = polygon_to_mask(src_img.shape, obj_data["polygon"])

            obj, mask_crop, crop_offset = extract_object(src_img, mask)

            if obj is None:
                continue

            obj, mask_crop = transform_object(obj, mask_crop)

            x, y = get_paste_position(target_img.shape, obj.shape)

            target_img = paste_object(target_img, obj, mask_crop, x, y)

            # 👉 TODO:
            # fix polygon transformation properly (hard part!)
            new_poly = update_polygon(obj_data["polygon"], (x, y), target_img.shape)

            new_labels.append(
                str(cls) + " " + " ".join(map(str, new_poly))
            )

        # =========================
        # 💾 SAVE RESULT
        # =========================

        name = f"cp_{idx:04d}"

        cv2.imwrite(os.path.join(out_img, name + ".jpg"), target_img)

        with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
            f.write("\n".join(new_labels))


if __name__ == "__main__":
    run_copy_paste()


# image = ski.io.imread("data/urban-waste.v5i.yolo26/train/images/pano_0000_000010_heading296-76_pitch-0-44_right90_square_fov90-0_640x640_jpg.rf.644c01dbdd5c2bd92f090a1a041fff52.jpg")
# edges = ski.filters.sobel(image)
# ski.io.imshow(image)
# ski.io.show()