import os

import cv2
import numpy as np
from distribution import get_label_distribution


import os
import cv2
import random

def read_yolo_labels(label_path: str):
    """Reads YOLO polygon labels into a list of [class_id, np.array([[x1,y1], [x2,y2]...])]."""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                # A valid polygon needs at least class + 3 sets of x,y coordinates (7 items)
                if len(parts) >= 7: 
                    cls_id = int(parts[0])
                    # Parse all remaining items into a 2D array of [x, y] coordinates
                    coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                    labels.append([cls_id, coords])
    return labels

def write_yolo_labels(labels: list, label_path: str):
    """Writes YOLO polygons back to a text file."""
    with open(label_path, 'w') as f:
        for cls_id, coords in labels:
            # Flatten the coordinates array back into a space-separated string
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
            f.write(f"{cls_id} {coords_str}\n")

def horizontal_flip(image, labels: list):
    """Flips the image horizontally and updates polygon points."""
    flipped_image = cv2.flip(image, 1)
    flipped_labels = []
    
    for cls_id, coords in labels:
        new_coords = coords.copy()
        # Invert only the X coordinates (1.0 - x)
        new_coords[:, 0] = 1.0 - new_coords[:, 0]
        flipped_labels.append([cls_id, new_coords])
        
    return flipped_image, flipped_labels

def zoom_crop_on_labels(image, labels: list, min_zoom=1.3, max_zoom=1.6):
    """Crops the image focusing on a polygon and resizes."""
    H, W = image.shape[:2]
    zoom_factor = random.uniform(min_zoom, max_zoom)
    nW, nH = int(W / zoom_factor), int(H / zoom_factor)
    
    # Determine crop center based on a random polygon's bounding box
    if labels:
        target_label = random.choice(labels)
        target_coords = target_label[1]
        cx_norm = (np.min(target_coords[:, 0]) + np.max(target_coords[:, 0])) / 2.0
        cy_norm = (np.min(target_coords[:, 1]) + np.max(target_coords[:, 1])) / 2.0
        cx_abs, cy_abs = int(cx_norm * W), int(cy_norm * H)
    else:
        cx_abs, cy_abs = W // 2, H // 2

    # Calculate top-left corner, clamping to image boundaries
    start_x = max(0, min(cx_abs - nW // 2, W - nW))
    start_y = max(0, min(cy_abs - nH // 2, H - nH))
    
    cropped_img = image[start_y:start_y + nH, start_x:start_x + nW]
    zoomed_img = cv2.resize(cropped_img, (W, H))
    
    zoomed_labels = []
    for cls_id, coords in labels:
        new_coords = coords.copy()
        
        # Convert to absolute pixels and shift relative to crop window
        new_coords[:, 0] = new_coords[:, 0] * W - start_x
        new_coords[:, 1] = new_coords[:, 1] * H - start_y
        
        # Check if the polygon is completely outside the new crop box
        if (np.max(new_coords[:, 0]) <= 0 or np.min(new_coords[:, 0]) >= nW or
            np.max(new_coords[:, 1]) <= 0 or np.min(new_coords[:, 1]) >= nH):
            continue 
            
        # Clamp points to the edges of the crop window
        new_coords[:, 0] = np.clip(new_coords[:, 0], 0, nW)
        new_coords[:, 1] = np.clip(new_coords[:, 1], 0, nH)
        
        # Normalize to the new window size
        new_coords[:, 0] /= nW
        new_coords[:, 1] /= nH
        
        zoomed_labels.append([cls_id, new_coords])
            
    return zoomed_img, zoomed_labels

def handle_img(image,
               labels: list,
               base_name: str,
               img_ext: str,
               image_out_folder: str,
               label_out_folder: str,
               copies_needed: int):
    """Generates augmented copies of an image and its YOLO labels."""
    
    # Ensure output directories exist
    os.makedirs(image_out_folder, exist_ok=True)
    os.makedirs(label_out_folder, exist_ok=True)
    
    for i in range(copies_needed):
        aug_img = image.copy()
        aug_labels = labels.copy()
        
        # 1. Apply zoom crop (focused on labels)
        aug_img, aug_labels = zoom_crop_on_labels(aug_img, aug_labels, 1.3, 1.6)
        
        # 2. Apply horizontal flip (50% probability to add variance across copies)
        if random.random() > 0.5:
            aug_img, aug_labels = horizontal_flip(aug_img, aug_labels)
            
        # Format output paths
        out_img_path = os.path.join(image_out_folder, f"{base_name}_aug_{i+1}{img_ext}")
        out_label_path = os.path.join(label_out_folder, f"{base_name}_aug_{i+1}.txt")
        
        # Save augmented image and labels
        cv2.imwrite(out_img_path, aug_img)
        write_yolo_labels(aug_labels, out_label_path)

def get_augmentation_counts(image_data: dict[str, dict[int, int]],
                            desired_additional_object_counts: dict[int, int],
                            max_copies):
    image_class_matrix = []
    fps = image_data.keys()
    for img in fps: # we have five classes
        image_class_matrix.append([image_data[img].get(cls, 0) for cls in range(5)])
    deficits = [desired_additional_object_counts.get(cls, 0) for cls in range(5)]

    counts = np.array(image_class_matrix, dtype=np.float32)
    deficits = np.array(deficits, dtype=np.float32)
    extra_copies = np.zeros(counts.shape[0], dtype=int)
    
    while True:
        # 2. Score every image based on current deficits
        scores = np.dot(counts, deficits)
        
        # 3. ENFORCE THE MAX COPY LIMIT
        # Any image that has reached max_copies gets a score of negative infinity,
        # completely disqualifying it from being selected again.
        scores[extra_copies >= max_copies] = -np.inf
        
        # 4. Find the image that helps our deficits the most
        best_idx = np.argmax(scores)
        max_score = scores[best_idx]
        
        # 5. Stop condition: 
        # If max_score <= 0, copying ANY remaining eligible image will do more harm than good.
        # Note: If ALL images hit max_copies, all scores become -inf, which is <= 0, 
        # so the loop safely breaks.
        if max_score <= 0:
            break
            
        # 6. "Copy" the image and update our deficits
        extra_copies[best_idx] += 1
        deficits -= counts[best_idx]
        
    return dict(zip(fps, [int(x) for x in extra_copies]))

def augment_classes(class_ratios: dict[int, float], input_folder: str, output_folder: str, max_copies_per_image):
    img_input = os.path.join(input_folder, "images")
    lbl_input = os.path.join(input_folder, "labels")
    img_output = os.path.join(output_folder, "images")
    lbl_output = os.path.join(output_folder, "labels")
    
    os.makedirs(img_output, exist_ok=True)
    os.makedirs(lbl_output, exist_ok=True)

    label_distribution, _, object_counts_per_image, *_ = get_label_distribution(lbl_input)
    desired_additional_object_counts: dict[int, int] = {cls: int(label_distribution.get(cls, 0) * (ratio - 1)) for cls, ratio in class_ratios.items()}
    copies_created: dict[int, int] = {cls: 0 for cls in class_ratios.keys()}

    total_additional_copies = sum(desired_additional_object_counts.values())
    print(f"Total additional copies to create: {total_additional_copies}")

    input_images = [f for f in os.listdir(img_input) if f.endswith('.jpg')]
    image_data = dict(zip(input_images, [dict(c) for c in object_counts_per_image]))
    augmentation_counts = get_augmentation_counts(image_data, desired_additional_object_counts, max_copies_per_image)
    for img_file, copy_count in augmentation_counts.items():
        base_name = os.path.splitext(img_file)[0]
        img_ext = os.path.splitext(img_file)[1]
        out_img_path = os.path.join(img_output, f"{base_name}_aug_0{img_ext}")
        out_label_path = os.path.join(lbl_output, f"{base_name}_aug_0.txt")
        img_path = os.path.join(img_input, img_file)
        label_path = os.path.join(lbl_input, img_file.replace('.jpg', '.txt'))

        image = cv2.imread(img_path)
        labels = read_yolo_labels(label_path)

        cv2.imwrite(out_img_path, image)
        write_yolo_labels(labels, out_label_path)

        if copy_count <= 0 or len(labels) == 0:
            continue
        
        handle_img(image, labels, base_name, img_ext, img_output, lbl_output, copy_count)
        for label in image_data[img_file]:
            copies_created[label] += copy_count * image_data[img_file][label]

    return copies_created