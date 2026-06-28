import io
import os
import json
from datetime import datetime
import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import time
import subprocess, gc

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
SAMPLE_SIZE = 100
SEED = 42  

# SHUTDOWN_URL = "http://localhost:8000/shutdown"

# Creates a dedicated folder for images inside your local directory
IMAGE_SAVE_DIR = "pope_evaluated_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

OUTPUT_FILE = f"pope_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

CHECKPOINT_FILE = "pope_checkpoint_partial.json"

REFRESH_URL = "http://localhost:8000/refresh"


def parse_sentence_to_binary(response_text: str) -> str:
    """Parses full-sentence VLM descriptions into a clean 'yes' or 'no'."""
    text = response_text.lower().strip()
    
    if text in ["yes", "yes.", "true", "correct"]:
        return "yes"
    if text in ["no", "no.", "false", "incorrect"]:
        return "no"
    
    negatives = [
        "there is no", "is not present", "doesn't have", 
        "cannot see", "there isn't", "not in the image"
    ]
    if any(neg in text for neg in negatives):
        return "no"
        
    words = text.split()
    if words:
        first_word = words[0].strip(".,!?")
        if first_word in ["yes", "yeah", "sure"]:
            return "yes"
        if first_word in ["no", "nope", "never"]:
            return "no"

    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
        
    return "yes"


def query_vlm(image: Image.Image, question: str) -> str:
    """Sends the image and question to the local FastAPI VLM."""
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
    
    


def evaluate_pope():
    print("Loading POPE dataset...")
    dataset = load_dataset("lmms-lab/POPE", split="test")

    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples to process: {total_samples}")

    # Restore status from checkpoint if it exists
    start_idx = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    total_valid = 0
    per_image_results = []

    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            start_idx = ckpt.get("processed_up_to", 0) + 1
            counts = ckpt.get("current_counts", {})
            tp = counts.get("tp", 0)
            fp = counts.get("fp", 0)
            tn = counts.get("tn", 0)
            fn = counts.get("fn", 0)
            total_valid = counts.get("total_valid", 0)
            per_image_results = ckpt.get("results", [])
            print(f"⏩ Found checkpoint! Resuming benchmark from index {start_idx}...")
        except Exception as e:
            print(f"[WARNING] Failed reading checkpoint ({e}). Starting fresh.")

    # Track how many items we have processed since our last fresh server boot
    batch_counter = 0 

    print(f"Starting full benchmark running on CPU...")

    # FIX 1: Cleaned up the nested tqdm bug
    for idx, sample in enumerate(tqdm(dataset)):

        if idx < start_idx:
            continue

        # --- RESTART TRIGGERS EVERY 500 SAMPLES ---
        if batch_counter >= 1000:
            try:
                ref_res = requests.post(REFRESH_URL, timeout=10)
                if ref_res.status_code == 200:
                    print("[BATCH] Server memory refreshed successfully.")
                else:
                    print(f"[WARNING] Server returned status {ref_res.status_code} during refresh.")
            except Exception as re:
                print(f"[WARNING] Could not hit refresh endpoint: {re}")
            
            batch_counter = 0 # Reset batch counter tracking

        real_id = str(sample.get("question_id", sample.get("id", f"sample_{idx}")))
        raw_question = sample["question"]
        question = raw_question + " Answer the question directly with 'Yes' or 'No' based strictly on clear visual evidence."
        ground_truth = sample["answer"].lower().strip()
        pil_image = sample["image"]

        # Ensure directory image footprint exists
        image_filename = f"{real_id}.jpg"
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        if not os.path.exists(image_path):
            pil_image.save(image_path, format="JPEG")

        vlm_response = query_vlm(pil_image, question)

        # Retries once if the server went down unexpectedly right before this sample
        if vlm_response == "error" or not vlm_response:
            print(f"\n[ERROR] Skipping index {idx} due to request failure.")
            continue

        total_valid += 1
        batch_counter += 1
        pred = parse_sentence_to_binary(vlm_response)

        # Classification verification matrix mapping
        match_category = "unknown"
        if ground_truth == "yes":
            if pred == "yes": tp += 1; match_category = "TP"
            else: fn += 1; match_category = "FN"
        elif ground_truth == "no":
            if pred == "no": tn += 1; match_category = "TN"
            else: fp += 1; match_category = "FP"

        per_image_results.append({
            "index": idx, "real_image_id": real_id, "local_image_path": image_path,
            "question": question, "ground_truth": ground_truth,
            "raw_model_response": vlm_response, "parsed_prediction": pred,
            "metric_type": match_category, "is_correct": pred == ground_truth
        })

        # Save standard checkpoints every 100 images for precision safety
        if total_valid % 100 == 0 and total_valid > 0:
            checkpoint_payload = {
                "processed_up_to": idx,
                "current_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total_valid": total_valid},
                "results": per_image_results
            }
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint_payload, f, indent=4)

    if total_valid == 0:
        print("No valid responses processed.")
        return
    
    # --- METRICS CALCULATION ---
    accuracy = (tp + tn) / total_valid
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    benchmark_report = {
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": {
            "total_evaluated": total_valid,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }
        },
        "detailed_predictions": per_image_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("       CUSTOM-ONNX-FASTVLM EVALUATION REPORT       ")
    print("=" * 50)
    print(f"Total Evaluated  : {total_valid}/{total_samples}")
    print("-" * 50)
    print(f"Accuracy         : {accuracy:.2%}")
    print(f"Precision        : {precision:.2%}")
    print(f"Recall           : {recall:.2%}")
    print(f"F1-Score         : {f1:.4f}")
    print("-" * 50)
    print(f"Confusion Matrix : TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Log JSON File    : {OUTPUT_FILE}")
    print(f"Saved Images Dir : ./{IMAGE_SAVE_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_pope()