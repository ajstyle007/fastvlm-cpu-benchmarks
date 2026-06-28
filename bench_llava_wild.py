import time
import requests
import io
import json
import numpy as np
from datasets import load_dataset
from PIL import Image

# --- CONFIGURATION ---
RESOLUTION = "1024"  
API_URL = "http://localhost:8000/predict"
OUTPUT_FILENAME = f"benchmark_results_{RESOLUTION}.json"

# 1. Load the small LLaVA-Bench dataset from Hugging Face
print("Loading LLaVA-Bench (In-the-Wild)...")
dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

# Performance containers
results_list = []

print(f"Loaded {len(dataset)} pairs. Starting live inference for {RESOLUTION} res...\n")

# 2. Iterate through the dataset
for idx, item in enumerate(dataset):
    question = item['question']
    pil_img = item['image']
    category = item.get('category', 'general')
    pair_id = item.get('question_id', f"q_{idx}")
    
    # Convert PIL Image to Bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    data = {"prompt": question}
    
    start_time = time.perf_counter()
    
    try:
        # stream=True is critical to capture the network TTFT mark accurately
        response = requests.post(API_URL, files=files, data=data, stream=True)
        
        first_token_received = False
        ttft_timestamp = 0.0
        
        # Read the streaming response chunk by chunk to isolate prompt evaluation delay
        for chunk in response.iter_content(chunk_size=1):
            if not first_token_received:
                ttft_timestamp = time.perf_counter()
                first_token_received = True
            pass 
            
        end_time = time.perf_counter()
        
        # Calculate Client-Side Metrics
        total_ttft = (ttft_timestamp - start_time) * 1000
        total_latency = (end_time - start_time) * 1000
        
        results_list.append({
            "id": pair_id,
            "category": category,
            "prompt_char_len": len(question),
            "client_ttft_ms": round(total_ttft, 2),
            "total_pipeline_ms": round(total_latency, 2)
        })
        
        print(f"[{idx+1}/{len(dataset)}] Category: {category}")
        print(f"     -> TTFT: {total_ttft:.2f} ms | Total: {total_latency:.2f} ms\n")
        
    except Exception as e:
        print(f"Error at index {idx}: {e}")
        continue

# 3. Calculate Final Stats & Save JSON
if results_list:
    ttfts = [r['client_ttft_ms'] for r in results_list]
    totals = [r['total_pipeline_ms'] for r in results_list]
    
    final_json_payload = {
        "metadata": {
            "target_resolution": f"{RESOLUTION}x{RESOLUTION}",
            "total_samples_evaluated": len(results_list),
            "summary_stats": {
                "avg_ttft_ms": round(float(np.mean(ttfts)), 2),
                "min_ttft_ms": round(float(np.min(ttfts)), 2),
                "max_ttft_ms": round(float(np.max(ttfts)), 2),
                "avg_total_pipeline_ms": round(float(np.mean(totals)), 2),
                "std_dev_ttft_ms": round(float(np.std(ttfts)), 2)
            }
        },
        "detailed_runs": results_list
    }
    
    # Write cleanly to disk
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as json_file:
        json.dump(final_json_payload, json_file, indent=4)
        
    print("==========================================================")
    print(f"✅ BENCHMARK RUN SAVED FOR {RESOLUTION} RESOLUTION")
    print(f"📊 Results saved successfully to: {OUTPUT_FILENAME}")
    print("==========================================================")