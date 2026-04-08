import json
import os
import shutil
import time
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

from data import get_image_fps

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE") 
OUTPUT_FOLDER = "data/gemini-labels/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
client = genai.Client(api_key=API_KEY)
MODEL_ID = 'gemini-flash-lite-latest'

PROMPT = """
Look closely and carefully at this street view image. I would like you to identify whether there is any waste in the street. Be rigorous and make sure to include all objects you can find.
Urban waste can be understood as "Any visible, discarded material or objects occupying public space (sidewalks, streets, tree grates). This strictly excludes permanent public infrastructure (like trash bins themselves - but does include waste deposited at or around them) and materials actively being carried or used by pedestrians."
The following types of waste are of interest
bulky waste
- "Large, discrete objects (furniture, appliances, mattresses) situated in public right-of-ways in an unorganized manner. Visual indicators include: being stacked haphazardly, missing essential parts (e.g., a three-legged chair), structural damage, or placement adjacent to public bins." (excludes furniture that is fulfilling a designated function such as tables or chairs of a restaurant)
garbage bag
- "Tied, knotted, or sealed plastic sacks resting on the ground. Primarily grey, black, or translucent yellow/blue (recycling). Excludes open shopping bags, backpacks, or bags actively held by humans."
card board
- "Cardboard material that exhibits visual signs of discard: flattened, torn, crushed, wet, bundled with twine, or mixed with other categories of waste. Strictly excludes pristine, taped, intact parcels resting directly at residential or commercial doorways."
litter
- "Small, uncontained, scattered debris. Visually identifiable as synthetic or heavily processed materials (plastic, foil, branded paper). Typically found resting in gutters, against curbs, or in tree grates. Examples: crushed cans, wrappers, loose plastic bags, cups."
other
- "Visible waste that defies the above categories but meets the general definition. Examples include: shattered glass, or scattered organic waste like dumped potting soil."
Please return only the types of waste you can identify in the image as a JSON array of strings. For example, if you see a cardboard box and a garbage bag, you would return ["cardboard", "garbage bag"]. If you don't see any waste, simply return an empty array [].
"""
if os.path.exists(os.path.join(OUTPUT_FOLDER, "labels.json")):
    with open(os.path.join(OUTPUT_FOLDER, "labels.json"), "r") as f:
        current_backup_state = json.load(f)
else:
    current_backup_state = {}
print(f"Loaded backup with {len(current_backup_state)} labeled images. Resuming from last state...")

def analyze_and_sort_images(image_filepaths):
    print(f"Starting analysis of {len(image_filepaths)} images...")
    labels: dict[str, list[str]] = {}
    
    for idx, filepath in enumerate(image_filepaths):
        filename = filepath
        if filename in current_backup_state:
            print(f"[{idx + 1}/{len(image_filepaths)}] Skipping {filename} (already labeled in backup).")
            labels[filename] = current_backup_state[filename]
            continue
        print(f"[{idx + 1}/{len(image_filepaths)}] Analyzing {filename}...")
        
        try:
            img = Image.open(filepath)
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[img, PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.45,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=1000,
                    ),
                )
            )
            
            answer = response.text.strip().upper()
            if "[" in answer and "]" in answer:
                if len(answer) == 2:
                    answer = []
                else:
                    answer = answer[1:-1].split(",")
                    answer = [a.strip().strip('"').strip("'") for a in answer]
                labels[filename] = answer
            else:
                labels[filename] = None
                
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
        
        if idx % 25 == 0:
            # store backup every 25 images
            with open(os.path.join(OUTPUT_FOLDER, "labels_backup.json"), "w") as f:
                json.dump(labels, f, indent=2)
    with open(os.path.join(OUTPUT_FOLDER, "labels_backup.json"), "w") as f:
        json.dump(labels, f, indent=2)

if __name__ == "__main__":
    fps = get_image_fps()
    valid_filepaths = [fp for fp, _, _, _ in fps if os.path.isfile(fp)]
    analyze_and_sort_images(valid_filepaths)