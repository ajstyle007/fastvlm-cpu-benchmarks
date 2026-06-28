import io
import os
import json
from datetime import datetime
import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
SAMPLE_SIZE = 100
SEED = 42  
# Creates a dedicated folder for images inside your local directory
IMAGE_SAVE_DIR = "pope_evaluated_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

OUTPUT_FILE = f"pope_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


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

    print(f"Shuffling dataset and extracting {SAMPLE_SIZE} random samples...")
    sampled_dataset = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    total_valid = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    per_image_results = []

    print(f"Starting benchmark running on CPU...")

    for idx, sample in enumerate(tqdm(sampled_dataset)):
        # Extracting the real underlying unique ID provided by POPE
        # If 'question_id' is an integer, we format it cleanly.
        real_id = str(sample.get("question_id", sample.get("id", f"sample_{idx}")))
        
        question = sample["question"]
        question = sample["question"] + " Answer the question directly with 'Yes' or 'No' based strictly on clear visual evidence."
        ground_truth = sample["answer"].lower().strip()
        pil_image = sample["image"]

        # --- SAVE IMAGE TO LOCAL DIRECTORY ---
        image_filename = f"{real_id}.jpg"
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        # Avoid resaving if it already exists from a previous run
        if not os.path.exists(image_path):
            pil_image.save(image_path, format="JPEG")

        vlm_response = query_vlm(pil_image, question)

        if vlm_response == "error" or not vlm_response:
            continue

        total_valid += 1
        pred = parse_sentence_to_binary(vlm_response)

        # Classification verification
        match_category = "unknown"
        if ground_truth == "yes":
            if pred == "yes":
                tp += 1
                match_category = "TP"
            else:
                fn += 1
                match_category = "FN"
        elif ground_truth == "no":
            if pred == "no":
                tn += 1
                match_category = "TN"
            else:
                fp += 1
                match_category = "FP"

        # Record log with real IDs and image local references
        per_image_results.append({
            "index": idx,
            "real_image_id": real_id,
            "local_image_path": image_path,
            "question": question,
            "ground_truth": ground_truth,
            "raw_model_response": vlm_response,
            "parsed_prediction": pred,
            "metric_type": match_category,
            "is_correct": pred == ground_truth
        })

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
    print(f"Total Evaluated  : {total_valid}/{SAMPLE_SIZE}")
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