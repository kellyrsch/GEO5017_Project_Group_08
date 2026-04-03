import os
import csv
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from data import get_image_fps

OUTPUT_CSV = "grounding_dino_scores.csv"

# Grounding DINO requires categories to be lowercase and separated by periods
TEXT_PROMPT = "litter . garbage bag . cardboard box . bulky waste . trash . debris . plastic . paper"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

processed_files = set()

if os.path.exists(OUTPUT_CSV):
    print(f"Found existing progress in {OUTPUT_CSV}. Loading...")
    try:
        # Read the existing CSV to find out what we've already done
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_files = set(df_existing['filename'].tolist())
        print(f"Resuming: Skipping {len(processed_files)} previously processed images.")
    except Exception as e:
        print(f"Could not read existing CSV: {e}. Starting fresh.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

all_files = [fp[0] for year, fps in get_image_fps().items() for fp in fps] # for now we only care about the file paths

# Filter out the files we've already processed
remaining_files = [f for f in all_files if f not in processed_files]

print(f"Found {len(all_files)} total images. {len(remaining_files)} remaining to process.")

# Initialize CSV with headers if we are starting from scratch
if not os.path.exists(OUTPUT_CSV) or len(processed_files) == 0:
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "max_confidence", "waste_objects_detected"])

# Open the CSV in 'append' mode
with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    index = -1
    for filename in tqdm(remaining_files):
        index += 1
        try:
            image = Image.open(filename).convert("RGB")
            inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=target_sizes
            )[0]
            
            scores = results["scores"].cpu().numpy()
            max_score = float(scores.max()) if len(scores) > 0 else 0.0
            object_count = len(scores)

            if object_count > 0:
                print(f"\n{filename}: Detected {object_count} waste objects with max confidence {max_score:.4f}")
            
            # Write the result immediately to the CSV
            writer.writerow([filename, max_score, object_count])
            
            if index % 10 == 0:  # Flush every 10 images
                # Flush forces the OS to write the buffer to disk right now
                # This is what guarantees no data loss if the terminal is killed
                f.flush() 
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

print("\nInference complete! Sorting results by confidence...")
try:
    df_final = pd.read_csv(OUTPUT_CSV)
    df_final = df_final.sort_values(by="max_confidence", ascending=False)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully sorted and saved {len(df_final)} results to {OUTPUT_CSV}")
except Exception as e:
    print(f"Error sorting the final CSV: {e}")