from collections import defaultdict
import os
import random
import skimage as ski
import cv2
import numpy as np

from distribution import get_label_distribution

def copy_object():
    pass

def get_img_objects(img, lines, remaining_objects_required_for_classes: dict[int, int]) -> list[tuple[int, list[float]]]:
    objects = []
    for idx, line in enumerate(lines):
        parts = line.split()
        label = int(parts[0])
        coords = list(map(float, parts[1:]))
        if label in remaining_objects_required_for_classes.keys() and remaining_objects_required_for_classes[label] > 0:
            objects.append((label, coords))
    return objects

def cut_objects(image, objects: list[tuple[int, list[float]]]):
    h, w = image.shape[:2]
    object_img_data = defaultdict(list)
    for label, coords in objects:
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

        # crop mask as well
        mask_crop = mask[y_min:y_max, x_min:x_max]

        # convert BGR → BGRA (adds alpha channel for transparent background)
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)

        # set alpha channel = mask
        cropped_rgba[:, :, 3] = mask_crop
        object_img_data[label].append(cropped_rgba)
    return [(label, img) for label, imgs in object_img_data.items() for img in imgs]

def paste_objects(canvas_img, objects_to_paste: list[tuple[int, np.ndarray]]):
    pass

def handle_img(img_path: str, labels_path: str, remaining_objects_required_for_classes: dict[int, int], images_to_paste_on: list[str]) -> tuple[dict[int, int], int]:
    copies_created_per_class = {cls: 0 for cls in remaining_objects_required_for_classes.keys()}

    image = cv2.imread(img_path)
    with open(labels_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    objects = get_img_objects(image, lines, remaining_objects_required_for_classes)
    if not objects:
        return copies_created_per_class, 0
    
    object_img_data = cut_objects(image, objects)    
    random.shuffle(object_img_data) # randomly distribute across images

    objects_per_image = max(1, len(object_img_data) // len(images_to_paste_on))

    current_image_objects = []
    current_image_index = 0
    for label, img in object_img_data:
        if copies_created_per_class[label] >= remaining_objects_required_for_classes[label]:
            continue

        if not images_to_paste_on:
            break

        current_image_objects.append(img)
        copies_created_per_class[label] += 1

        if len(current_image_objects) >= objects_per_image:
            canvas_img_path = images_to_paste_on[current_image_index]
            canvas_img = cv2.imread(canvas_img_path)
            paste_objects(canvas_img, current_image_objects)
            current_image_objects = []
            current_image_index += 1

    return copies_created_per_class, current_image_index + 1

def augment_classes(class_ratios: dict[int, float], input_folder: str, output_folder: str):
    img_input = os.path.join(input_folder, "images")
    lbl_input = os.path.join(input_folder, "labels")
    img_output = os.path.join(output_folder, "images")
    lbl_output = os.path.join(output_folder, "labels")
    
    os.makedirs(img_output, exist_ok=True)
    os.makedirs(lbl_output, exist_ok=True)

    canvas_images = [os.path.join(img_input, f) for f in os.listdir(img_input) if f.endswith('.jpg')]

    label_distribution, *_ = get_label_distribution(lbl_input)
    desired_additional_object_counts: dict[int, int] = {cls: int(label_distribution.get(cls, 0) * (ratio - 1)) for cls, ratio in class_ratios.items()}
    copies_created: dict[int, int] = {cls: 0 for cls in class_ratios.keys()}

    total_additional_copies = sum(desired_additional_object_counts.values())
    print(f"Total additional copies to create: {total_additional_copies}")

    objects_per_canvas_image = (total_additional_copies // len(canvas_images) + 1) * 1.5 # we might not fill up every image completely (this is a rule of thumb - could be implemted cleaner)
    print(f"Objects to paste per canvas image: {objects_per_canvas_image}")

    input_images = [f for f in os.listdir(img_input) if f.endswith('.jpg')]
    for img_file in input_images:
        img_path = os.path.join(img_input, img_file)
        lbl_path = os.path.join(lbl_input, img_file.replace('.jpg', '.txt'))
        
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        classes_in_image = set(int(line.split()[0]) for line in lines)
        classes_to_copy = [cls for cls in classes_in_image if cls in desired_additional_object_counts and copies_created[cls] < desired_additional_object_counts[cls]]
        if not classes_to_copy:
            continue

        images_to_paste_on = random.sample(canvas_images, min(int(objects_per_canvas_image), len(canvas_images)))
        remaining_objects_required_for_classes = {cls: desired_additional_object_counts[cls] - copies_created[cls] for cls in classes_to_copy}
        copies_created_per_class, images_consumed = handle_img(img_path, lbl_path, remaining_objects_required_for_classes, images_to_paste_on)
        for cls, count in copies_created_per_class.items():
            copies_created[cls] += count

if __name__ == "__main__":
    input_folder = r"data\urban-waste.v5i.yolo26\train"
    output_folder = r"data\urban-waste.v5i.yolo26\train_augmented"
    class_ratios = {
        0: 3,  # bulky waste
        1: 1.5,  # cardboard
        3: 2,  # litter
        4: 5,  # other
    }
    augment_classes(class_ratios, input_folder, output_folder)