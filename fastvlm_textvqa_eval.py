import io
import os
import json
import glob
from datetime import datetime
import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import re

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
REFRESH_URL = "http://localhost:8000/refresh"
MAX_SAMPLES = None  # Set to an integer to limit testing, or None for all images

LOCAL_DATA_DIR = "textvqa_data"       # Folder containing your parquet files
IMAGE_SOURCE_DIR = "failed_vqa_images"  # Source folder where your images live

OUTPUT_FILE = f"textvqa_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_1024.json"
CHECKPOINT_FILE = "textvqa_checkpoint_partial_1024.json"


# --- VQA ACCURACY SCORING ---
def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vqa_accuracy(predicted: str, ground_truths: list) -> float:
    pred_norm = normalize_answer(predicted)
    match_count = sum(1 for gt in ground_truths if normalize_answer(gt) == pred_norm)
    return min(match_count / 3.0, 1.0)


# --- API CALL ---
def query_vlm(image: Image.Image, question: str) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    files = {"image": ("image.jpg", img_byte_arr, "image/jpeg")}
    data = {"prompt": question}
    try:
        response = requests.post(API_URL, files=files, data=data, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"\nAPI Error ({response.status_code}): {response.text}")
            return "error"
    except Exception as e:
        print(f"\nRequest failed: {e}")
        return "error"


# --- MAIN EVALUATION ---
def evaluate_textvqa_from_local_images():
    # 1. Dynamically scan and build an index mapping from the parquet files
    print("Loading TextVQA metadata from local parquet files...")
    local_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.parquet"))
    
    if not local_files:
        print(f"[ERROR] No parquet files found in {LOCAL_DATA_DIR}. Exiting.")
        return
    
    dataset = load_dataset(
        "parquet",
        data_files={"validation": local_files},
        split="validation"
    )
    
    metadata_map = {}
    for row in tqdm(dataset, desc="Indexing parquet text data", leave=False):
        q_id = str(row["question_id"])
        metadata_map[q_id] = {
            "question": row["question"],
            "answers": row["answers"]
        }
    
    # 2. Gather images directly from the local folder
    print(f"\nScanning source folder: '{IMAGE_SOURCE_DIR}'...")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_image_files = [f for f in os.listdir(IMAGE_SOURCE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not all_image_files:
        print(f"No images found in ./{IMAGE_SOURCE_DIR}/. Exiting.")
        return

    if MAX_SAMPLES:
        all_image_files = all_image_files[:MAX_SAMPLES]

    total_samples = len(all_image_files)
    print(f"Total local images found to evaluate: {total_samples}")

    # --- INITIALIZE / RESTORE TRACKING METRICS ---
    start_idx = 0
    total_valid = 0
    total_vqa_score = 0.0
    exact_matches = 0
    per_image_results = []
    batch_counter = 0

    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            start_idx = ckpt.get("processed_up_to", 0) + 1
            counts = ckpt.get("current_counts", {})
            total_valid = counts.get("total_valid", 0)
            total_vqa_score = counts.get("total_vqa_score", 0.0)
            exact_matches = counts.get("exact_matches", 0)
            per_image_results = ckpt.get("results", [])
            batch_counter = total_valid  # Match the count for memory refresh logic
            print(f"⏩ Checkpoint found! Resuming from index {start_idx} (Processed: {total_valid})...")
        except Exception as e:
            print(f"[WARNING] Failed reading checkpoint ({e}). Starting fresh from scratch.")

    print("Starting TextVQA benchmark evaluation...\n")

    for idx, img_filename in enumerate(tqdm(all_image_files, total=total_samples)):

        if idx < start_idx:
            continue

        # --- MEMORY REFRESH EVERY 1000 SAMPLES ---
        if batch_counter > 0 and batch_counter % 1000 == 0:
            try:
                ref_res = requests.post(REFRESH_URL, timeout=10)
                if ref_res.status_code == 200:
                    print("\n[BATCH] Server memory refreshed successfully.")
                else:
                    print(f"\n[WARNING] Refresh returned status {ref_res.status_code}.")
            except Exception as re_err:
                print(f"\n[WARNING] Could not hit refresh endpoint: {re_err}")

        # --- EXTRACT FIELDS & MATCH WITH INDEX ---
        image_path = os.path.join(IMAGE_SOURCE_DIR, img_filename)
        question_id = str(os.path.splitext(img_filename)[0])
        
        meta = metadata_map.get(question_id)
        if not meta:
            print(f"\n[WARNING] Question ID {question_id} not found in parquet metadata. Skipping.")
            continue

        question_raw = meta["question"]
        ground_truths = meta["answers"]
        question = question_raw + " Answer concisely with just the word or short phrase visible in the image."

        # --- LOAD IMAGE ---
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as img_err:
            print(f"\n[ERROR] Failed to load image {image_path}: {img_err}. Skipping.")
            continue

        # --- QUERY MODEL ---
        vlm_response = query_vlm(pil_image, question)

        if vlm_response == "error" or not vlm_response:
            print(f"\n[ERROR] Skipping image {img_filename} due to request failure.")
            continue

        total_valid += 1
        batch_counter += 1

        # --- SCORE ---
        score = vqa_accuracy(vlm_response, ground_truths)
        total_vqa_score += score

        pred_norm = normalize_answer(vlm_response)
        is_exact = any(normalize_answer(gt) == pred_norm for gt in ground_truths)
        if is_exact:
            exact_matches += 1

        per_image_results.append({
            "index": idx,
            "question_id": question_id,
            "local_image_path": image_path,
            "question": question_raw,
            "ground_truths": ground_truths,
            "raw_model_response": vlm_response,
            "normalized_prediction": pred_norm,
            "vqa_score": round(score, 4),
            "is_exact_match": is_exact
        })

        # --- SAVE SAFE CHECKPOINT DURING THE RUN ---
        if total_valid % 10 == 0:
            checkpoint_payload = {
                "processed_up_to": idx,
                "current_counts": {
                    "total_valid": total_valid,
                    "total_vqa_score": total_vqa_score,
                    "exact_matches": exact_matches
                },
                "results": per_image_results
            }
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint_payload, f, indent=4)
            tqdm.write(f"[Checkpoint] Saved at index {idx} | Running VQA Acc: {total_vqa_score/total_valid:.2%}")

    if total_valid == 0:
        print("No valid responses processed.")
        return

    # --- FINAL METRICS ---
    avg_vqa_accuracy = total_vqa_score / total_valid
    exact_match_rate = exact_matches / total_valid

    benchmark_report = {
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": {
            "total_evaluated": total_valid,
            "total_samples_in_dataset": total_samples,
            "vqa_accuracy": round(avg_vqa_accuracy, 4),
            "exact_match_rate": round(exact_match_rate, 4),
            "exact_matches": exact_matches,
        },
        "detailed_predictions": per_image_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=4, ensure_ascii=False)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print("\n" + "=" * 50)
    print("     CUSTOM-ONNX-FASTVLM TEXTVQA EVALUATION REPORT   ")
    print("=" * 50)
    print(f"Total Evaluated  : {total_valid}/{total_samples}")
    print("-" * 50)
    print(f"VQA Accuracy     : {avg_vqa_accuracy:.2%}   ← Primary Metric")
    print(f"Exact Match Rate : {exact_match_rate:.2%}   ← Secondary Metric")
    print(f"Exact Matches    : {exact_matches}/{total_valid}")
    print("-" * 50)
    print(f"Log JSON File    : {OUTPUT_FILE}")
    print(f"Source Images Dir: ./{IMAGE_SOURCE_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_textvqa_from_local_images()