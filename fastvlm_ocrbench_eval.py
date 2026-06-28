import io
import os
import json
from datetime import datetime
import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import re, ast

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
REFRESH_URL = "http://localhost:8000/refresh"
MAX_SAMPLES = None  # Set to e.g. 500 to limit, or None for all available samples

LOCAL_DATA_DIR = "OCRbench_data"   # ← your local folder with the OCRBench parquet files
IMAGE_SAVE_DIR = "ocrbench_evaluated_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

OUTPUT_FILE = f"ocrbench_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
CHECKPOINT_FILE = "ocrbench_checkpoint_partial_new.json"


# --- OCRBENCH ACCURACY SCORING ---
def normalize_answer(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def try_parse_dict(text: str) -> dict:
    """Safely attempts to parse text into a dictionary regardless of quotes."""
    if not text or not isinstance(text, str):
        return None
    text_stripped = text.strip()
    if not (text_stripped.startswith('{') and text_stripped.endswith('}')):
        return None
    try:
        return json.loads(text_stripped)
    except Exception:
        try:
            # Fallback for python-style string dicts with single quotes: {'key': 'val'}
            return ast.literal_eval(text_stripped)
        except Exception:
            return None


def ocrbench_accuracy(predicted: str, ground_truths: list) -> float:
    """
    Advanced hybrid evaluator supporting word-boundary text matching 
    and structural dictionary key-value parsing for complex OCR tasks.
    """
    pred_norm = normalize_answer(predicted)
    if not pred_norm:
        return 0.0

    for gt in ground_truths:
        gt_str = str(gt).strip()
        
        # --- 1. TRY DICTIONARY VALUE EVALUATION ---
        gt_dict = try_parse_dict(gt_str)
        pred_dict = try_parse_dict(predicted)
        
        if gt_dict:
            # Extract target values to see if model captured them anywhere in its output
            target_values = [normalize_answer(str(v)) for v in gt_dict.values() if v and v != "###"]
            if not target_values:
                continue
                
            # If the model produced a valid JSON block, check its contents structural-wise
            if pred_dict:
                matched_vals = 0
                pred_values_norm = [normalize_answer(str(v)) for v in pred_dict.values()]
                for tv in target_values:
                    if any(tv in pv for pv in pred_values_norm):
                        matched_vals += 1
                # Give points if it hit a majority of the structural data fields
                if matched_vals / len(target_values) >= 0.5:
                    return 1.0
            
            # If model failed to write valid JSON but dumped the raw values textually
            matched_raw_vals = sum(1 for tv in target_values if tv in pred_norm)
            if matched_raw_vals / len(target_values) >= 0.5:
                return 1.0

        # --- 2. FALLBACK TO STICKY WORD BOUNDARY MATCHING ---
        gt_norm = normalize_answer(gt_str)
        if not gt_norm:
            continue
            
        pattern = r'\b' + re.escape(gt_norm) + r'\b'
        if re.search(pattern, pred_norm):
            return 1.0
            
    return 0.0


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
def evaluate_ocrbench():
    print("Loading OCRBench from local parquet files...")

    # Listing the first 3 parquet files verified in your terminal output
    local_files = [
        os.path.join(LOCAL_DATA_DIR, "test-00000-of-00011.parquet"),
        os.path.join(LOCAL_DATA_DIR, "test-00001-of-00011.parquet"),
        os.path.join(LOCAL_DATA_DIR, "test-00002-of-00011.parquet"),
    ]

    # Verify files exist before loading to catch pathway typos early
    missing_files = [f for f in local_files if not os.path.exists(f)]
    if missing_files:
        print(f"[ERROR] Could not find local parquet files: {missing_files}")
        return

    # Load directly from local files
    dataset = load_dataset(
        "parquet",
        data_files={"test": local_files},
        split="test"
    )

    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples to evaluate: {total_samples}")

    # --- RESTORE FROM CHECKPOINT ---
    start_idx = 0
    total_valid = 0
    total_ocr_score = 0.0
    per_image_results = []

    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            start_idx = ckpt.get("processed_up_to", 0) + 1
            counts = ckpt.get("current_counts", {})
            total_valid = counts.get("total_valid", 0)
            total_ocr_score = counts.get("total_ocr_score", 0.0)
            per_image_results = ckpt.get("results", [])
            print(f"⏩ Checkpoint found! Resuming from index {start_idx}...")
        except Exception as e:
            print(f"[WARNING] Failed reading checkpoint ({e}). Starting fresh.")

    batch_counter = 0
    print("Starting OCRBench evaluation...\n")

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
        # OCRBench fields map closely to standard datasets but handle strings/lists natively
        question_id = str(sample.get("question_id", f"ocr_{idx}"))
        question_raw = sample["question"]
        
        # Ensure answers format is handled cleanly as a list
        ground_truths = sample.get("answers")
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        elif not ground_truths:
            ground_truths = [sample.get("answer", "")]

        pil_image = sample["image"]
        # question = question_raw + " Answer concisely with just the text or short phrase visible in the image."

        if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', question_raw):
            question = question_raw + " 仅输出答案，使用与问题相同的键，格式为JSON。"
        else:
            question = question_raw + " Answer concisely with just the text or short phrase visible in the image."


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
        score = ocrbench_accuracy(vlm_response, ground_truths)
        total_ocr_score += score
        pred_norm = normalize_answer(vlm_response)

        per_image_results.append({
            "index": idx,
            "question_id": question_id,
            "local_image_path": image_path,
            "question": question_raw,
            "ground_truths": ground_truths,
            "raw_model_response": vlm_response,
            "normalized_prediction": pred_norm,
            "score": int(score)
        })

        # --- CHECKPOINT EVERY 100 VALID SAMPLES ---
        if total_valid % 10 == 0:
            checkpoint_payload = {
                "processed_up_to": idx,
                "current_counts": {
                    "total_valid": total_valid,
                    "total_ocr_score": total_ocr_score
                },
                "results": per_image_results
            }
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint_payload, f, indent=4)
            tqdm.write(f"[Checkpoint] Saved at index {idx} | "
                       f"Running OCR Accuracy: {total_ocr_score/total_valid:.2%}")

    if total_valid == 0:
        print("No valid responses processed.")
        return

    # --- FINAL METRICS ---
    avg_ocr_accuracy = total_ocr_score / total_valid

    benchmark_report = {
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": {
            "total_evaluated": total_valid,
            "total_samples_in_subset": total_samples,
            "ocr_accuracy": round(avg_ocr_accuracy, 4),
        },
        "detailed_predictions": per_image_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=4, ensure_ascii=False)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\n[INFO] Checkpoint file removed after successful completion.")

    print("\n" + "=" * 50)
    print("      CUSTOM-ONNX-FASTVLM OCRBENCH REPORT       ")
    print("=" * 50)
    print(f"Total Evaluated  : {total_valid}/{total_samples}")
    print("-" * 50)
    print(f"OCR Accuracy     : {avg_ocr_accuracy:.2%}   ← Core Match Score")
    print("-" * 50)
    print(f"Log JSON File    : {OUTPUT_FILE}")
    print(f"Saved Images Dir : ./{IMAGE_SAVE_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_ocrbench()