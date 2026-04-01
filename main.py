import requests
import time

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

start_time = time.time()

# Took code from: https://huggingface.co/docs/transformers/model_doc/grounding-dino

model_id = "IDEA-Research/grounding-dino-tiny"
device = Accelerator().device

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

#image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_url = "GEO5017-Project-UrbanWaste/UrbanWaste-images-10k-right/year_2016/TMX7316010203-000188/pano_0000_002298_heading220.50_pitch-0.22_right90_square_fov90.0_640x640.jpg"
#image = Image.open(requests.get(image_url, stream=True).raw)
image = Image.open(image_url)
# Check for cats and remote controls
text_labels = [["a bicycle", "a building", "a trash bag", "a car", "litter", "paper"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.4,
    target_sizes=[image.size[::-1]]
)

result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    draw = ImageDraw.Draw(image)

    # Optional: nicer font (falls back if not found)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        box = box.tolist()

        # Draw rectangle
        draw.rectangle(box, outline="red", width=3)

        # Prepare label text
        text = f"{label} ({round(score.item(), 2)})"

        # Position text slightly above the box
        text_position = (box[0], box[1] - 20)

        # Draw text
        draw.text(text_position, text, fill="red", font=font)

    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
    image.show()

print("Runtime: ", time.time() - start_time)