import os
import random
import shutil

from basic_augmentation import augment_classes as basic_augment_classes
from copy_paste_augmentation import augment_classes as copy_paste_augment_classes
from data import get_image_fps


def get_all_image_names_and_labels(data_folder: str):
    img_folder = os.path.join(data_folder, "images")
    lbl_folder = os.path.join(data_folder, "labels")

    image_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    res = []
    for img_file in image_files:
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(lbl_folder, label_file)
        if not os.path.exists(label_path):
            continue
        res.append((os.path.join(img_folder, img_file), label_path))
    return res

def clean_up_file_name(file_name: str):
    base, ext = os.path.splitext(file_name)
    clean_base = base
    #clean_base = ".".join(base.split(".")[:-1]) if "." in base else base
    if clean_base.find(".rf") != -1:
        clean_base = clean_base[:clean_base.find(".rf")]
    if clean_base.startswith("year"):
        pano_start_idx = clean_base.find("pano")
        if pano_start_idx != -1:
            clean_base = clean_base[pano_start_idx:]
    if clean_base.endswith("_jpg.rf"):
        clean_base = clean_base[:-7]
    if clean_base.endswith("_jpg"):
        clean_base = clean_base[:-4]
    if "-" in clean_base:
        clean_base = clean_base.replace("-", ".")
    if clean_base.find("pitch") != -1:
        pitch_idx = clean_base.find("pitch")
        if clean_base[pitch_idx+5:pitch_idx+6] == ".":
            clean_base = clean_base[:pitch_idx+5] + clean_base[pitch_idx+6:]
    return clean_base + ext

def cleanup_filenames(data_folder: str):
    image_label_pairs = get_all_image_names_and_labels(data_folder)
    
    for img_path, label_path in image_label_pairs:
        clean_file_name = clean_up_file_name(os.path.basename(img_path))
        img_dir = os.path.dirname(img_path)
        label_dir = os.path.dirname(label_path)
        
        new_img_path = os.path.join(img_dir, clean_file_name)
        new_label_path = os.path.join(label_dir, clean_file_name.replace('.jpg', '.txt'))
        
        shutil.move(img_path, new_img_path)
        shutil.move(label_path, new_label_path)

def get_file_names_for_years(years: list[int]) -> list[str]:
    source_fps = get_image_fps()
    fps = [fp for fp, year, *_ in source_fps if year in years]
    return [clean_up_file_name(os.path.basename(fp))[:-4] for fp in fps]

if __name__ == "__main__":
    #build_training_dataset()
    cleanup_filenames(r"data\urban-waste.v5i.yolo26-basic-aug-v3\test")