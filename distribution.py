import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import csv

labels_folder = r"data\urban-waste.v5i.yolo26-testing\train_augmented\labels"

def get_label_distribution(labels_folder: str):
    total_objects_per_label = Counter()
    objects_per_image = []
    object_counts_per_image = []

    images_with_same_label = 0
    images_with_mixed_labels = 0

    co_occurrence = defaultdict(int)

    label_sizes = defaultdict(list)
    label_centers_x = defaultdict(list)
    label_centers_y = defaultdict(list)
    for file in os.listdir(labels_folder):
        if not file.endswith(".txt"):
            continue

        with open(os.path.join(labels_folder, file), "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        labels = []

        for line in lines:
            parts = line.split()
            label = int(parts[0])
            coords = list(map(float, parts[1:]))

            labels.append(label)

            # polygon → bbox
            xs = coords[0::2]
            ys = coords[1::2]

            if len(xs) == 0:
                continue

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            width = xmax - xmin
            height = ymax - ymin
            area = width * height

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2

            label_sizes[label].append(area)
            label_centers_x[label].append(cx)
            label_centers_y[label].append(cy)

        #per-image stats
        num_objects = len(labels)
        objects_per_image.append(num_objects)
        object_counts_per_image.append(Counter(labels))

        unique_labels = set(labels)

        if len(unique_labels) == 1 and num_objects > 1:
            images_with_same_label += 1
        elif len(unique_labels) > 1:
            images_with_mixed_labels += 1

        #co-occurrence
        unique_list = list(unique_labels)
        for i in range(len(unique_list)):
            for j in range(i + 1, len(unique_list)):
                pair = tuple(sorted((unique_list[i], unique_list[j])))
                co_occurrence[pair] += 1

        total_objects_per_label.update(labels)
    return total_objects_per_label, objects_per_image, object_counts_per_image, images_with_same_label, images_with_mixed_labels, label_sizes, label_centers_x, label_centers_y, co_occurrence

if __name__ == "__main__":
    total_objects_per_label, objects_per_image, object_counts_per_image, images_with_same_label, images_with_mixed_labels, label_sizes, label_centers_x, label_centers_y, co_occurrence = get_label_distribution(labels_folder)

    output = "\nBasic Stats:\n"
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
    
    print(output)

    # #objects per image
    # plt.figure()
    # plt.hist(objects_per_image, bins=20)
    # plt.title("Objects per Image")
    # plt.xlabel("Objects per image")
    # plt.ylabel("Frequency")
    # plt.show()

    # #label distribution
    # plt.figure()
    # labels = list(total_objects_per_label.keys())
    # counts = list(total_objects_per_label.values())
    # plt.bar(labels, counts)
    # plt.title("Label Distribution")
    # plt.xlabel("Label")
    # plt.ylabel("Count")
    # plt.show()

    # #object size distribution
    # all_sizes = [s for sizes in label_sizes.values() for s in sizes]
    # plt.figure()
    # plt.hist(all_sizes, bins=30)
    # plt.title("Object Size Distribution")
    # plt.xlabel("BBox Area")
    # plt.ylabel("Frequency")
    # plt.show()

    # #size per label
    # plt.figure()
    # data = [label_sizes[l] for l in sorted(label_sizes.keys())]
    # plt.boxplot(data)
    # plt.title("Object Size per Label")
    # plt.xlabel("Label index")
    # plt.ylabel("Area")
    # plt.show()

    # #co-occurrence
    # top_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10]
    # pairs = [str(p[0]) for p in top_pairs]
    # counts = [p[1] for p in top_pairs]

    # plt.figure()
    # plt.bar(pairs, counts)
    # plt.title("Top Co-occurring Labels")
    # plt.xlabel("Label pairs")
    # plt.ylabel("Count")
    # plt.xticks(rotation=45)
    # plt.show()

    # with open("label_distribution.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["label", "count"])
    #     for label, count in total_objects_per_label.items():
    #         writer.writerow([label, count])
    #
    # print("\nSaved: label_distribution.csv")