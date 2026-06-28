import json
import os

CHECKPOINT_FILE = "gqa_checkpoint_partial.json"
FINAL_REPORT_FILE = "gqa_final_benchmark_report.json"

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 1. Grab your exact stopped counts
    counts = data.get("current_counts", {})
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    tn = counts.get("tn", 0)
    fn = counts.get("fn", 0)
    total_valid = counts.get("total_valid", 0)
    
    # 2. Compute accurate metrics based on the equations
    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 3. Structure the clean summary metrics payload
    summary_metrics = {
        "total_evaluated": total_valid,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
    }
    
    # 4. Inject into the dictionary structure
    final_output = {
        "processed_up_to": data.get("processed_up_to"),
        "summary_metrics": summary_metrics,
        "current_counts": counts,
        "results": data.get("results", [])
    }
    
    # 5. Write the final patched report file cleanly
    with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Successfully created {FINAL_REPORT_FILE} with injected metrics block!")
    print(json.dumps(summary_metrics, indent=4))
else:
    print(f"❌ Error: Could not find {CHECKPOINT_FILE}")