import os
os.environ["YOLO_AUTOINSTALL"] = "False"

from ultralytics import YOLOWorld

# 1. Load a pretrained YOLO-World model (it will download automatically)
# Options: yolov8s-world.pt (fastest), yolov8m-world.pt, yolov8l-world.pt (most accurate)
model = YOLOWorld('yolov8s-world.pt')

# 2. Define the exact classes you want to detect via text prompt
model.set_classes(["garbage bin"])

# 3. Run inference on a single image or a whole directory of streetview images
# You can pass a list of paths, a directory path, or a single image path
image_source_folder = r'data\UrbanWaste-images-10k-right\year_2017\TMX7316010203-000227\pano_0000_002484_heading35.79_pitch-0.37_right90_square_fov90.0_640x640.jpg' 
output_label_folder = r""

for file in os.listdir(image_source_folder):
    if not file.endswith(".jpg"):
        continue
    image_source = os.path.join(image_source_folder, file)
    results = model(image_source)

    # 4. Extract and store the bounding boxes
    for result in results:
        print(f"--- Results for {result.path} ---")
        boxes = result.boxes
        
        for box in boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            
            # Get bounding box coordinates in [x1, y1, x2, y2] format
            bbox = box.xyxy[0].tolist() 
            
            print(f"Found: {class_name} (Confidence: {confidence:.2f})")
            print(f"Bounding Box: {bbox}\n")