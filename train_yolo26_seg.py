YAML_PATH = r"data\urban-waste.v5i.yolo26-basic-aug-v3\data.yaml"
AVAILABLE_MODELS = {
    "nano": "yolo26n-seg.pt",
    "small": "yolo26s-seg.pt",
    "medium": "yolo26m-seg.pt",
    "large": "yolo26l-seg.pt",
    "extra_large": "yolo26x-seg.pt"
}

SELECTED_MODEL = "large"

if __name__ == "__main__":
    print(f"Downloading model {SELECTED_MODEL}...")
    from ultralytics import YOLO
    model = YOLO(AVAILABLE_MODELS[SELECTED_MODEL])

    print(f"\nStarting training on data: {YAML_PATH}")
    try:
        results = model.train(
            data=YAML_PATH,
            epochs=200,             # we will most likely stop due to patience way earlier anway
            imgsz=640,
            batch=16,               # Reduce if you hit Out of Memory (OOM) errors on GPU
            device='0', # '0' for GPU, or 'cpu' if no GPU is available
            project='yolo26_seg',
            name=f'run_{SELECTED_MODEL}_v3',
            task='segment',

            patience=25,            # Early Stopping: If accuracy doesn't improve for 15 epochs, stop training early to prevent overfitting
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