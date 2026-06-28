import io
import os
import json
from datetime import datetime
import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import re

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
REFRESH_URL = "http://localhost:8000/refresh"
MAX_SAMPLES = None  # Set to e.g. 500 to limit, or None for all 5000

LOCAL_DATA_DIR = "textvqa_data"   # ← your local folder with the 3 parquet files
IMAGE_SAVE_DIR = "failed_vqa_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

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
def evaluate_textvqa():
    print("Loading TextVQA from local parquet files...")

    local_files = [
        os.path.join(LOCAL_DATA_DIR, "validation-00000-of-00003.parquet"),
        os.path.join(LOCAL_DATA_DIR, "validation-00001-of-00003.parquet"),
        os.path.join(LOCAL_DATA_DIR, "validation-00002-of-00003.parquet"),
    ]

    # Load directly from local files — no internet needed
    dataset = load_dataset(
        "parquet",
        data_files={"validation": local_files},
        split="validation"
    )

    if MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples to evaluate: {total_samples}")

    # --- RESTORE FROM CHECKPOINT ---
    start_idx = 0
    total_valid = 0
    total_vqa_score = 0.0
    exact_matches = 0
    per_image_results = []

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
            print(f"⏩ Checkpoint found! Resuming from index {start_idx}...")
        except Exception as e:
            print(f"[WARNING] Failed reading checkpoint ({e}). Starting fresh.")

    batch_counter = 0
    print("Starting TextVQA benchmark...\n")

    for idx, sample in enumerate(tqdm(dataset, total=total_samples)):

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

        # --- EXTRACT SAMPLE FIELDS ---
        question_id = str(sample.get("question_id", f"sample_{idx}"))
        question_raw = sample["question"]
        ground_truths = sample["answers"]
        pil_image = sample["image"]

        question = question_raw + " Answer concisely with just the word or short phrase visible in the image."

        # --- SAVE IMAGE LOCALLY ---
        image_filename = f"{question_id}.jpg"
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        if not os.path.exists(image_path):
            pil_image.save(image_path, format="JPEG")

        # --- QUERY MODEL ---
        vlm_response = query_vlm(pil_image, question)

        if vlm_response == "error" or not vlm_response:
            print(f"\n[ERROR] Skipping index {idx} due to request failure.")
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

        # --- CHECKPOINT EVERY 100 VALID SAMPLES ---
        if total_valid % 100 == 0:
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
            tqdm.write(f"[Checkpoint] Saved at index {idx} | "
                       f"Running VQA Acc: {total_vqa_score/total_valid:.2%}")

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
        print("\n[INFO] Checkpoint file removed after successful completion.")

    print("\n" + "=" * 50)
    print("    CUSTOM-ONNX-FASTVLM TEXTVQA EVALUATION REPORT   ")
    print("=" * 50)
    print(f"Total Evaluated  : {total_valid}/{total_samples}")
    print("-" * 50)
    print(f"VQA Accuracy     : {avg_vqa_accuracy:.2%}   ← Primary Metric")
    print(f"Exact Match Rate : {exact_match_rate:.2%}   ← Secondary Metric")
    print(f"Exact Matches    : {exact_matches}/{total_valid}")
    print("-" * 50)
    print(f"Log JSON File    : {OUTPUT_FILE}")
    print(f"Saved Images Dir : ./{IMAGE_SAVE_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_textvqa()