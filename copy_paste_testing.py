import os
import cv2
import random
import numpy as np
import skimage as ski

# setup for later to apply this script to our data
images_folder = "data/urban-waste.v5i.yolo26/train/images"
labels_folder = "data/urban-waste.v5i.yolo26/train/labels"

out_img = "data/urban-waste.v5i.yolo26/train/aug_copypaste_images"
out_lbl = "data/urban-waste.v5i.yolo26/train/aug_copypaste_labels"

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_lbl, exist_ok=True)


# define minority classes that should be copy/pasted (basically everything except for garbage bags)
minority_classes = [0, 1, 3, 4]

image_path = "data/urban-waste.v5i.yolo26/train/images/pano_0000_000034_heading302-25_pitch-0-48_right90_square_fov90-0_640x640_jpg.rf.c6eadd3785eb18d10975e22eac5fa7bf.jpg"
labels_file_path = "data/urban-waste.v5i.yolo26/train/labels/pano_0000_000034_heading302-25_pitch-0-48_right90_square_fov90-0_640x640_jpg.rf.c6eadd3785eb18d10975e22eac5fa7bf.txt"
test_paste_image_path = "data/urban-waste.v5i.yolo26/train/cropped_objects/pano_0000_000218_heading64.56_pitch-4.48_right90_square_fov90.0_640x640.jpg"

# output folder
output_dir = "data/urban-waste.v5i.yolo26/train/cropped_objects"
os.makedirs(output_dir, exist_ok=True)

# read image and lines of the labels file
image = cv2.imread(image_path)
h, w = image.shape[:2]
with open(labels_file_path) as f:
    lines = [l.strip() for l in f if l.strip()]

#labels_file = open(labels_file_path)
#lines = [l.strip() for l in labels_file if l.strip()]
# print("lines", lines)


## crop object from image and save as seperate file
for idx, line in enumerate(lines):
    parts = line.split()
    label = int(parts[0])
    coords = list(map(float, parts[1:]))

    if label not in minority_classes:
        continue

    print(f"Processing object {idx}, class {label}")

    # convert to pixel coords
    pts = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * w)
        y = int(coords[i+1] * h)
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32)

    # create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # extract object
    extracted = cv2.bitwise_and(image, image, mask=mask)

    # crop to bbox
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        continue

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = extracted[y_min:y_max, x_min:x_max]

    # save output
    # crop mask as well
    mask_crop = mask[y_min:y_max, x_min:x_max]

    # convert BGR → BGRA (adds alpha channel for transparent background)
    cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)

    # set alpha channel = mask
    cropped_rgba[:, :, 3] = mask_crop

    out_path = os.path.join(output_dir, f"obj_{idx}_class_{label}.png")
    cv2.imwrite(out_path, cropped_rgba)

    print("Saved with transparency:", out_path)


## paste cropped object to image
paste_to_image = cv2.imread(test_paste_image_path)
cropped_object = cv2.imread("data/urban-waste.v5i.yolo26/train/cropped_objects/obj_7_class_0.png", cv2.IMREAD_UNCHANGED)

# split object channels
obj_rgb = cropped_object[:, :, :3]
alpha = cropped_object[:, :, 3] / 255.0  # normalize to 0–1

H, W = paste_to_image.shape[:2]
h, w = obj_rgb.shape[:2]

# ensure object fits
if h > H or w > W:
    print("Object too large!")
    exit()

# x anywhere
x = np.random.randint(0, W - w)

# y only in lower 1/4
y_min = int(0.75 * H)
y_max = H - h

y = np.random.randint(y_min, max(y_min + 1, y_max))

# extract ROI
roi = paste_to_image[y:y+h, x:x+w]

# blend to make edges less sharp and thus making the pasting look more realistic
for c in range(3):
    roi[:, :, c] = (alpha * obj_rgb[:, :, c] +
                    (1 - alpha) * roi[:, :, c])

paste_to_image[y:y+h, x:x+w] = roi

# save new image file
cv2.imwrite("data/urban-waste.v5i.yolo26/train/cropped_objects/test_paste_to_image.jpg", paste_to_image)