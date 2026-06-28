import json
from datetime import datetime

def merge_and_update_evals(base_5000_path, corrected_2902_path, output_path):
    # 1. Load both JSON files
    with open(base_5000_path, 'r', encoding='utf-8') as f:
        data_5000 = json.load(f)
        
    with open(corrected_2902_path, 'r', encoding='utf-8') as f:
        data_2902 = json.load(f)

    # 2. Map the corrected records by question_id for O(1) lookups
    corrected_map = {
        str(pred["question_id"]): pred 
        for pred in data_2902["detailed_predictions"]
    }

    # 3. Update the 5000-record list
    updated_predictions = []
    updated_count = 0

    for pred in data_5000["detailed_predictions"]:
        q_id = str(pred["question_id"])
        
        if q_id in corrected_map:
            # Replace with the corrected record
            updated_predictions.append(corrected_map[q_id])
            updated_count += 1
        else:
            # Keep the original record if no correction exists
            updated_predictions.append(pred)

    print(f"Successfully updated {updated_count} records out of 5000.")

    # 4. Recalculate summary metrics dynamically based on updated data
    total_evaluated = len(updated_predictions)
    
    # Calculate exact matches
    exact_matches = sum(1 for pred in updated_predictions if pred.get("is_exact_match") is True)
    exact_match_rate = round(exact_matches / total_evaluated, 4) if total_evaluated > 0 else 0.0
    
    # Calculate VQA Accuracy (average of all vqa_scores)
    vqa_scores = [pred.get("vqa_score", 0.0) for pred in updated_predictions]
    vqa_accuracy = round(sum(vqa_scores) / total_evaluated, 4) if total_evaluated > 0 else 0.0

    # 5. Build final updated JSON structure
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": {
            "total_evaluated": total_evaluated,
            "total_samples_in_dataset": data_5000["summary_metrics"].get("total_samples_in_dataset", total_evaluated),
            "vqa_accuracy": vqa_accuracy,
            "exact_match_rate": exact_match_rate,
            "exact_matches": exact_matches
        },
        "detailed_predictions": updated_predictions
    }

    # 6. Save to new file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"Saved the updated dataset to: {output_path}")


merge_and_update_evals("textvqa_benchmark_20260624_235116.json", "textvqa_benchmark_20260628_005523_1024.json", "updated_file_textvqa.json")