import glob
import os
from pathlib import Path

from ultralytics import YOLO

BENCHMARKS_FP = r"data\benchmarks\P@100"
MAIN_TEST_SET = r"data\generated-datasets\uw-basic-aug-v4\test"

def find_top_waste_images(model_path, test_dir, output_dir, top_x=10):
    """
    Ranks images based on the highest confidence segmentation prediction.
    
    Args:
        model_path (str): Path to your trained YOLO segmentation weights (e.g., 'best.pt').
        test_dir (str): Path to the folder containing test images.
        output_dir (str): Path to save the top images with drawn polygons and labels.
        top_x (int): How many top images to return.
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    image_scores = []

    print(f"Streaming inference directly from {test_dir}...")

    results = model.predict(
        source=test_dir, 
        stream=True, 
        imgsz=640, 
        half=True, 
        verbose=False
    )
    
    for result in results:
        # result.path automatically contains the original image path
        path = result.path 
        max_confidence = 0.0
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                confidence = float(box.conf[0].item())
                if confidence > max_confidence:
                    max_confidence = confidence
                        
        image_scores.append((path, max_confidence))
        
    image_scores.sort(key=lambda x: x[1], reverse=True)
    top_images = image_scores[:top_x]
    
    print(f"\n--- Top {top_x} Images Most Likely to Contain Waste ---")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (path, score) in enumerate(top_images, 1):
        filename = os.path.basename(path)
        print(f"{i}. {filename} - Confidence: {score:.4f}")
        
        # Run inference again to get annotated image
        result = model.predict(source=path, verbose=False)[0]
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"{i:03d}_{filename}")
        result.save(filename=output_path)
        
    # save to file
    with open(os.path.join(output_dir, "top_waste_images.txt"), "w") as f:
        for path, score in top_images:
            f.write(f"{path}\t{score:.4f}\n")
    return top_images

def evaluate_top_results(top_images, labels_dir):
    """
    Evaluates how many of the top ranked images actually contain waste based on ground truth labels.
    
    Args:
        top_images (list): List of tuples [(image_path, confidence), ...] from the previous function.
        labels_dir (str): Path to the directory containing the ground truth .txt label files.
        waste_class_ids (list, optional): List of class IDs that count as waste. If None, any label counts.
        
    Returns:
        dict: Contains the total checked, total correct, and the accuracy percentage.
    """
    if not top_images:
        print("No images to evaluate.")
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}

    correct_count = 0
    total_checked = len(top_images)
    
    print(f"\n--- Evaluating Top {total_checked} Results ---")
    
    for i, (img_path, confidence) in enumerate(top_images, 1):
        filename = os.path.basename(img_path)
        # Swap the image extension for .txt to find the label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        has_waste = False
        
        # Check if the label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                has_waste = lines is not None and len(lines) > 0
        else:
            has_waste = False
        
        if has_waste:
            correct_count += 1
            status = "✅ TRUE POSITIVE (Waste found in ground truth)"
        else:
            status = "❌ FALSE POSITIVE (No waste in ground truth)"
            
        print(f"{i}. {filename} (Conf: {confidence:.2f}) -> {status}")
        
    # Calculate final metrics
    accuracy = (correct_count / total_checked) * 100
    
    print("\n--- Final Evaluation ---")
    print(f"Total Images Checked: {total_checked}")
    print(f"Actually Contained Waste: {correct_count}")
    print(f"Precision @ {total_checked}: {accuracy:.2f}%")
    
    return {
        'total': total_checked,
        'correct': correct_count,
        'accuracy': accuracy
    }

def run_yolo_benchmark(model_path, test_dir, output_dir):
    model = YOLO(model_path)
    print("Starting benchmark on test set...")
    metrics = model.val(
        data=os.path.join(Path(test_dir).parent.parent, "data.yml"),
        split='test',
        batch=16,
        imgsz=640,
        plots=False
    )

    results_str = "\n--- Benchmark Results ---\n"
    results_str += f"Mean Average Precision @ IoU=0.50 (mAP50): {metrics.box.map50:.4f}\n"
    results_str += f"Mean Average Precision @ IoU=0.50:0.95 (mAP50-95): {metrics.box.map:.4f}\n"
    results_str += f"Precision: {metrics.box.mp:.4f}\n"
    results_str += f"Recall: {metrics.box.mr:.4f}\n"

    results_str += "\n--- Class-Specific mAP50-95 ---\n"
    for i, class_id in enumerate(metrics.box.ap_class_index):
        class_name = model.names[class_id]
        class_map = metrics.box.maps[i]
        results_str += f"{class_name}: {class_map:.4f}\n"
        
    speeds = metrics.speed
    results_str += "\n--- Inference Speed ---\n"
    results_str += f"Pre-process: {speeds['preprocess']:.2f} ms/img\n"
    results_str += f"Inference:   {speeds['inference']:.2f} ms/img\n"
    results_str += f"Post-process:{speeds['postprocess']:.2f} ms/img\n"
    
    with open(os.path.join(output_dir, "benchmark_results.txt"), "w") as f:
        f.write(results_str)

def benchmark_model(model_name: str, model_path: str):
    test_images_dir = os.path.join(MAIN_TEST_SET, "images")
    test_labels_dir = os.path.join(MAIN_TEST_SET, "labels")
    out_path = os.path.join(BENCHMARKS_FP, model_name)
    #top_images = find_top_waste_images(model_path, test_images_dir, out_path, top_x=100)
    with open(os.path.join(out_path, "top_waste_images.txt")) as f:
        top_images = []
        for line in f:
            path, conf = line.strip().split("\t")
            top_images.append((path, float(conf)))
    result = evaluate_top_results(top_images, test_labels_dir)
    run_yolo_benchmark(model_path, test_images_dir, out_path)

if __name__ == "__main__":
    model_weights = r"runs\segment\yolo26_seg\run_small_v4\weights\best.pt"
    benchmark_model("v4-small", model_weights)
