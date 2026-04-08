from collections import defaultdict
import os
import random
import shutil

import numpy as np

from cleanup_roboflow_augmentations import clean_up_file_name, cleanup_filenames, get_file_names_for_years
from basic_augmentation import augment_classes as basic_augment_classes
from copy_paste_augmentation import augment_classes as copy_paste_augment_classes
from data import get_image_fps
from distribution import get_label_distribution

GROUND_TRUTH_FOLDER = r"data\hand_labelled_ground_truth"
OUTPUT_FOLDER = r"data\generated-datasets"

def load_all_data():
    available_data = defaultdict(lambda: {"hand-labelled": [], "background": []})
    # available data structure: {year: {"hand-labelled" | "background": list[(name, (source_image_path, source_label_path))]}
    source_fps_with_metadata = get_image_fps()
    clean_source_fps_with_metadata = {clean_up_file_name(os.path.basename(fp)): (fp, year, *rest) for fp, year, *rest in source_fps_with_metadata}
    ground_truth_labelled_data = {}
    for fp in os.listdir(os.path.join(GROUND_TRUTH_FOLDER, "images")):
        if fp.endswith(".jpg"):
            clean_fp = clean_up_file_name(fp)
            full_fp = os.path.join(GROUND_TRUTH_FOLDER, "images", fp)
            ground_truth_labelled_data[clean_fp] = (full_fp, os.path.join(GROUND_TRUTH_FOLDER, "labels", fp.replace('.jpg', '.txt')))
    no_label_data = {clean_fp: clean_source_fps_with_metadata[clean_fp] for clean_fp in clean_source_fps_with_metadata if clean_fp not in ground_truth_labelled_data}
    for clean_fp, (image_fp, label_fp) in ground_truth_labelled_data.items():
        metadata = clean_source_fps_with_metadata.get(clean_fp)
        if metadata is None:    
            raise ValueError(f"Warning: No metadata found for {clean_fp}. Skipping.")
        year = metadata[1]
        available_data[year]["hand-labelled"].append((clean_fp, (image_fp, label_fp)))
    for clean_fp, (image_fp, year, *_) in no_label_data.items():
        available_data[year]["background"].append((clean_fp, (image_fp, None)))
    for year in sorted(available_data.keys()):
        hand_labelled_percent = len(available_data[year]["hand-labelled"]) / (len(available_data[year]["hand-labelled"]) + len(available_data[year]["background"])) * 100 if (len(available_data[year]["hand-labelled"]) + len(available_data[year]["background"])) > 0 else 0
        print(f"Year {year}: {len(available_data[year]['hand-labelled'])} hand-labelled images ({hand_labelled_percent:.2f}%), {len(available_data[year]['background'])} background images ({100 - hand_labelled_percent:.2f}%)")
    return available_data

def save_img(name: str, image_fp: str, label_fp: str | None, dest_folder: str):
    img_dest = os.path.join(dest_folder, "images", name)
    lbl_dest = os.path.join(dest_folder, "labels", name.replace('.jpg', '.txt'))

    shutil.copy(image_fp, img_dest)
    if label_fp is not None:
        shutil.copy(label_fp, lbl_dest)
    else:
        with open(lbl_dest, 'w') as f:
            pass # create empty label file

def percentage_split(all_data: dict[str, list[tuple[str, str | None]]], background_samples_percent: float, train_percent: float, valid_percent: float, test_percent: float):
    background_data = all_data["background"]
    labelled_data = all_data["hand-labelled"]

    if background_samples_percent is not None:
        num_background_samples = int(len(labelled_data) * background_samples_percent)
        background_data = random.sample(background_data, min(num_background_samples, len(background_data)))
        print(f"Using {len(background_data)} background samples out of {len(all_data['background'])} available ({background_samples_percent*100:.2f}%)")
    
    all_data_combined = labelled_data + background_data
    random.shuffle(all_data_combined)
    total = len(all_data_combined)
    train_end = int(total * train_percent)
    valid_end = train_end + int(total * valid_percent)
    training_data = all_data_combined[:train_end]
    validation_data = all_data_combined[train_end:valid_end]
    test_data = all_data_combined[valid_end:]

    split_data = {
        'train': training_data,
        'valid': validation_data,
        'test': test_data
    }
    data_summary = {
        'train': {
            'hand-labelled': len([d for d in training_data if d in labelled_data]),
            'background': len([d for d in training_data if d in background_data])
        },
        'valid': {
            'hand-labelled': len([d for d in validation_data if d in labelled_data]),
            'background': len([d for d in validation_data if d in background_data])
        },
        'test': {
            'hand-labelled': len([d for d in test_data if d in labelled_data]),
            'background': len([d for d in test_data if d in background_data])
        }
    }
    return split_data, data_summary

def year_split(all_data: dict[int, dict[str, list[tuple[str, str | None]]]], background_samples_percent: float, train_years: list[int], valid_years: list[int], test_years: list[int]):
    training_labelled_data = [d for year in train_years for d in all_data[year]["hand-labelled"]]
    validation_labelled_data = [d for year in valid_years for d in all_data[year]["hand-labelled"]]
    test_data = [d for year in test_years for d in all_data[year]["hand-labelled"] + all_data[year]["background"]] # test data always includes all images

    training_background_data = [d for year in train_years for d in all_data[year]["background"]]
    num_background_samples = int(len(training_labelled_data) * background_samples_percent)
    training_background_data = random.sample(training_background_data, min(num_background_samples, len(training_background_data)))

    validation_background_data = [d for year in valid_years for d in all_data[year]["background"]]
    num_validation_background_samples = int(len(validation_labelled_data) * background_samples_percent)
    validation_background_data = random.sample(validation_background_data, min(num_validation_background_samples, len(validation_background_data)))

    split_data = {
        'train': training_labelled_data + training_background_data,
        'valid': validation_labelled_data + validation_background_data,
        'test': test_data
    }

    data_summary = {
        'train': {
            'hand-labelled': len(training_labelled_data),
            'background': len(training_background_data)
        },
        'valid': {
            'hand-labelled': len(validation_labelled_data),
            'background': len(validation_background_data)
        },
        'test': {
            'hand-labelled': len([d for d in test_data if d in [d for year in test_years for d in all_data[year]["hand-labelled"]]]),
            'background': len([d for d in test_data if d in [d for year in test_years for d in all_data[year]["background"]]])
        }
    }
    return split_data, data_summary

def setup_dataset_folders(name: str):
    output_path = os.path.join(OUTPUT_FOLDER, name)
    os.makedirs(os.path.join(output_path, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test", "labels"), exist_ok=True)
    return output_path

def apply_augmentation(output_path: str, augmentation_ratios: dict[int, float], max_augments_per_image: int):
    augmented_training_path = os.path.join(output_path, "train_augmented")
    os.makedirs(os.path.join(augmented_training_path, "images"), exist_ok=True)

    basic_augmentation = {}
    copy_paste_augmentation = {}
    for cls, ratio in augmentation_ratios.items():
        if ratio <= 3 or True: # don't do copy/paste augmentation for now
            basic_augmentation[cls] = ratio
        else:
            basic_augmentation[cls] = 3
            copy_paste_augmentation[cls] = ratio - 3

    basic_copies_created = basic_augment_classes(basic_augmentation,
                                                    os.path.join(output_path, "train"),
                                                    augmented_training_path,
                                                    max_augments_per_image)
    
    copy_paste_copies_created = defaultdict(int)
    if copy_paste_augmentation:
        copy_paste_copies_created = copy_paste_augment_classes(copy_paste_augmentation,
                                                                os.path.join(output_path, "train"),
                                                                augmented_training_path)
        
    return basic_copies_created, copy_paste_copies_created

def create_metadata(output_path: str, split_summary: dict, basic_copies_created: dict[int, int], copy_paste_copies_created: dict[int, int]):
    output = f"Dataset Summary for {output_path}:\n\n"

    output += "Data Split:\n"
    for split, summary in split_summary.items():
        total = summary['hand-labelled'] + summary['background']
        hand_labelled_percent = summary['hand-labelled'] / total * 100 if total > 0 else 0
        background_percent = summary['background'] / total * 100 if total > 0 else 0
        output += f"{split.capitalize()}: {total} images ({summary['hand-labelled']} containing waste ({hand_labelled_percent:.2f}%), {summary['background']} background ({background_percent:.2f}%))\n"

    if basic_copies_created or copy_paste_copies_created:
        output += "\nAugmentation Summary:\n"
        if basic_copies_created:
            output += "Steps applied: Basic Augmentation\n\n 1. zoom in by 1.3x to 1.6x (random distribution) on a random object\n\n 2. random horizontal flip\n\n"
            output += "This resulted in the following extra objects created per Class:\n"
            for cls, count in basic_copies_created.items():
                output += f"Class {cls}: {count} extra objects\n"
        if copy_paste_copies_created:
            output += "Steps applied: Copy-Paste Augmentation\n\n 1. Cut out desired objects from images\n\n 2. Paste them onto background images - preferably by a garbage-bin or on a sidewalk\n\n"
            output += "This resulted in the following extra objects created per Class:\n"
            for cls, count in copy_paste_copies_created.items():
                output += f"Class {cls}: {count} extra objects\n"

    train_labels_folder = os.path.join(output_path, "train", "labels")
    total_objects_per_label, objects_per_image, object_counts_per_image, images_with_same_label, images_with_mixed_labels, label_sizes, label_centers_x, label_centers_y, co_occurrence = get_label_distribution(train_labels_folder)

    output += "\nBasic Stats for Training Data:\n"
    output += f"Total images: {len(objects_per_image)}\n"
    output += f"Avg objects/image: {np.mean(objects_per_image):.2f}\n"
    output += f"Min objects/image: {np.min(objects_per_image)}\n"
    output += f"Max objects/image: {np.max(objects_per_image)}\n"

    output += "\nLabel Counts:\n"
    for label, count in sorted(total_objects_per_label.items()):
        output += f"Label {label}: {count}\n"

    output += "\nImage Types:\n"
    output += f"Single-object images: {sum(1 for x in objects_per_image if x == 1)}\n"
    output += f"Multi-object images: {sum(1 for x in objects_per_image if x > 1)}\n"
    output += f"Same-label images: {images_with_same_label}\n"
    output += f"Mixed-label images: {images_with_mixed_labels}\n"

    output += "\nSize Stats:\n"
    for label in sorted(label_sizes.keys()):
        sizes = label_sizes[label]
        output += f"Label {label}: mean={np.mean(sizes):.4f}, min={np.min(sizes):.4f}, max={np.max(sizes):.4f}\n"

    output += "\nPosition Stats:\n"
    for label in sorted(label_centers_x.keys()):
        output += f"Label {label}: mean_x={np.mean(label_centers_x[label]):.3f}, mean_y={np.mean(label_centers_y[label]):.3f}\n"

    output += "\nTop Co-occurrences:\n"
    sorted_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_pairs[:10]:
        output += f"{pair}: {count}\n"

    metadata_fp = os.path.join(output_path, "metadata.txt")
    with open(metadata_fp, 'w') as f:
        f.write(output)

def create_data_yml(output_path: str, has_augmentation: bool):
    data_yml_content = f"""train: ../{'train_augmented' if has_augmentation else 'train'}/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['bulky waste', 'cardboard', 'garbage bag', 'litter', 'other']
"""
    data_yml_fp = os.path.join(output_path, "data.yml")
    with open(data_yml_fp, 'w') as f:
        f.write(data_yml_content)

def build_dataset(name: str,
                  background_samples_percent: float,
                  year_train_split: dict[int, str] | None = None,
                  train_valid_test_split: tuple[float, float, float] | None = None,
                  augmentation_ratios: dict[int, float] = {},
                  max_augments_per_image: int = 5):
    if not ((year_train_split is None) != (train_valid_test_split is None)):
        raise ValueError("Exactly one of year_train_split or train_valid_test_split must be provided.")
    
    output_path = setup_dataset_folders(name)

    available_data = load_all_data()
    
    if train_valid_test_split is not None:
        all_data = {"hand-labelled": [], "background": []}
        for year in available_data:
            all_data["hand-labelled"] += available_data[year]["hand-labelled"]
            all_data["background"] += available_data[year]["background"]
        data_split, split_summary = percentage_split(all_data, background_samples_percent, *train_valid_test_split)
    elif year_train_split is not None:
        assert set(year_train_split.values()) == {"train", "valid", "test"}, "year_train_split values must be 'train', 'valid', or 'test'"
        data_split, split_summary = year_split(available_data, 
                                background_samples_percent,
                                train_years=[year for year, split in year_train_split.items() if split == "train"],
                                valid_years=[year for year, split in year_train_split.items() if split == "valid"],
                                test_years=[year for year, split in year_train_split.items() if split == "test"])
            
    for split, data in data_split.items():
        for name, (image_fp, label_fp) in data:
            save_img(name, image_fp, label_fp, os.path.join(output_path, split))

    basic_copies_created = {}
    copy_paste_copies_created = {}
    if augmentation_ratios:
        basic_copies_created, copy_paste_copies_created = apply_augmentation(output_path, augmentation_ratios, max_augments_per_image)

    create_metadata(output_path, split_summary, basic_copies_created, copy_paste_copies_created)
    create_data_yml(output_path, len(augmentation_ratios.keys()) > 0)

if __name__ == "__main__":
    build_dataset(
        "uw-basic-aug-v4",
        background_samples_percent=0.66,
        year_train_split={
            2016: "train",
            2017: "train",
            2018: "train",
            2019: "train",
            2020: "valid",
            2021: "valid",
            2022: "test",
            2023: "test"
        },
        augmentation_ratios={
            0: 8,  # bulky waste
            1: 4,  # cardboard
            2: 1.5,  # garbage bag (we want to reduce this class)
            3: 3,  # litter
            4: 12,  # other
        },
        max_augments_per_image=5
    )