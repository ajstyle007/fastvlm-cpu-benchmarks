import os
import json
import shutil

# --- CONFIGURATION ---
JSON_FILE_PATH = "textvqa_benchmark_20260624_235116.json"  
SOURCE_DIR = "textvqa_evaluated_images"
TARGET_DIR = "failed_vqa_images"  

# Create the target folder if it doesn't already exist
os.makedirs(TARGET_DIR, exist_ok=True)

# --- PROCESSING ---
try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find the JSON file at '{JSON_FILE_PATH}'")
    exit(1)

predictions = data.get("detailed_predictions", [])
copied_count = 0
missing_count = 0

print(f"Starting extraction for vqa_score == 0.0...")

for pred in predictions:
    # Check if the score is exactly 0.0
    if pred.get("vqa_score") == 0.0:
        question_id = pred.get("question_id")
        
        # Build the expected image filename (e.g., "34602.jpg")
        image_name = f"{question_id}.jpg"
        src_path = os.path.join(SOURCE_DIR, image_name)
        dest_path = os.path.join(TARGET_DIR, image_name)
        
        # Verify the source image exists before copying
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            print(f"Warning: Image not found in source folder: {src_path}")
            missing_count += 1

print("\n--- Summary ---")
print(f"Successfully copied: {copied_count} images to '{TARGET_DIR}'")
if missing_count > 0:
    print(f"Missing from source folder: {missing_count} images")