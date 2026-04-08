import os


AVAILABLE_MODELS = {
    "nano": "yolo26n-seg.pt",
    "small": "yolo26s-seg.pt",
    "medium": "yolo26m-seg.pt",
    "large": "yolo26l-seg.pt",
    "extra_large": "yolo26x-seg.pt"
}

def train_model(name: str, model_size: str, yaml_path: str, epochs: int, patience: int, batch_size: int):
    selected_model = AVAILABLE_MODELS.get(model_size)
    if not selected_model:
        print(f"Error: Model size '{model_size}' is not valid. Please choose from: {list(AVAILABLE_MODELS.keys())}")
        return
    print(f"Downloading model {selected_model}...")
    from ultralytics import YOLO
    model = YOLO(selected_model)

    print(f"\nStarting training on data: {yaml_path}")
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,             # we will most likely stop due to patience way earlier anway
            imgsz=640,
            batch=batch_size,               # Reduce if you hit Out of Memory (OOM) errors on GPU
            device='0', # '0' for GPU, or 'cpu' if no GPU is available
            project='yolo26_seg',
            name=name,
            task='segment',

            patience=patience,            # Early Stopping: If accuracy doesn't improve for 15 epochs, stop training early to prevent overfitting
            save_period=50,         # Save a backup checkpoint of the model every 10 epochs
            mask_ratio=4,

            # make sure the model doesn't apply its own augmentations
            hsv_h=0.0,              
            hsv_s=0.0,              
            hsv_v=0.0,              
            degrees=0.0,          
            translate=0.0,        
            scale=0.0,            
            shear=0.0,             
            perspective=0.0,      
            flipud=0.0,           
            fliplr=0.0,           
            mosaic=0.0,          
            mixup=0.0,      
        )
        print("\nTraining completed successfully! Your final weights are saved in the 'yolo26_seg' directory.")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

    model_output_dir = r"runs\segment\yolo26_seg"
    return os.path.join(model_output_dir, name)