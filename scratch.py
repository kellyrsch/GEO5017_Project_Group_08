import json
from collections import defaultdict, Counter
from itertools import combinations
import shutil

from data import get_image_fps, get_manually_labelled_filenames

def analyze_json_strings(filepath):
    # Load the JSON data
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    if total_items == 0:
        print("The JSON file is empty.")
        return

    # Trackers
    # string_presence_count: counts how many lists (items) contain a specific string
    string_presence_count = Counter()
    
    # co_occurrence: tracks how often string A appears in the same list as string B
    co_occurrence = defaultdict(Counter)

    for key, string_list in data.items():
        # Convert to a set to ignore duplicate strings within the same list 
        # (we only care IF it appears in the item, not how many times)
        unique_strings = set(string_list)

        # 1. Update overall presence count
        for s in unique_strings:
            string_presence_count[s] += 1

        # 2. Update co-occurrence
        # itertools.combinations gets all unique pairs in the set
        for s1, s2 in combinations(unique_strings, 2):
            co_occurrence[s1][s2] += 1
            co_occurrence[s2][s1] += 1

    # Output the results
    print(f"Total items (keys) processed: {total_items}\n")
    print("-" * 50)

    # Sort strings by their overall frequency for a cleaner report
    for s, count in string_presence_count.most_common():
        percentage = (count / total_items) * 100
        
        print(f"String: '{s}'")
        print(f"  - Present in: {percentage:.2f}% of items ({count}/{total_items})")
        
        # Get the top 3 strings it most commonly appears with
        top_co_occurring = co_occurrence[s].most_common(3)
        
        if top_co_occurring:
            print("  - Most commonly appears with:")
            for co_str, co_count in top_co_occurring:
                print(f"      * '{co_str}' ({co_count} times)")
        else:
            print("  - Never appears alongside other strings.")
            
        print("-" * 50)

def get_images_with_waste(label_fp):
    with open(label_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images_with_waste = []
    for key, string_list in data.items():
        if len(string_list) > 0:
            images_with_waste.append(key)
    
    return images_with_waste

def get_images_without_waste(label_fp):
    with open(label_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images_without_waste = []
    for key, string_list in data.items():
        if len(string_list) == 0:
            images_without_waste.append(key)
    
    return images_without_waste

if __name__ == "__main__":
    fps = get_images_without_waste(r"data\gemini-labels\labels.json")
    files = {fp.split("\\")[-1] for fp in fps}
    manual_labelled_files = set(get_manually_labelled_filenames())
    files_without_manual_labels = files - manual_labelled_files
    filepaths_without_manual_labels = [fp for fp in fps if fp.split("\\")[-1] in files_without_manual_labels]
    destination_folder = r"data\manual_labelling_subset\no_labels"
    for fp in filepaths_without_manual_labels:
        filename = "_".join(fp.split("\\")[-3:])
        new_fp = f"{destination_folder}\\{filename}"
        # copy image
        shutil.copy(fp, new_fp)