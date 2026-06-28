import gc
import atexit
from datasets import load_dataset

print("--- CHECKING TEXTVQA ---")
try:
    textvqa_stream = load_dataset("lmms-lab/TextVQA", split="validation", streaming=True)
    sample_vqa = next(iter(textvqa_stream))
    print("TextVQA Keys:", sample_vqa.keys())
    print("TextVQA Question Example:", sample_vqa.get("question"))
    print("TextVQA Answer Example:", sample_vqa.get("answers"))
except Exception as e:
    print("TextVQA load error:", e)

print("\n--- CHECKING OCRBENCH ---")

# These are the correct/active OCRBench repos on HuggingFace as of mid-2025
OCR_CANDIDATES = [
    "echo840/OCRBench",           # most common mirror
    "HuggingFaceM4/OCRBench",
    "THUDM/OCRBench",
]

ocr_loaded = False
for repo in OCR_CANDIDATES:
    try:
        print(f"Trying: {repo} ...")
        ocr_stream = load_dataset(repo, split="test", streaming=True)
        ocr_iter = iter(ocr_stream)
        sample_ocr = next(ocr_iter)

        print(f"✅ Loaded from: {repo}")
        print("OCRBench Keys:", list(sample_ocr.keys()))
        print("OCRBench Question Example:", sample_ocr.get("question"))
        print("OCRBench Answer Example:", sample_ocr.get("answers") or sample_ocr.get("answer"))

        del sample_ocr
        del ocr_iter
        gc.collect()
        ocr_loaded = True
        break

    except Exception as e:
        print(f"  ✗ Failed: {e}")

if not ocr_loaded:
    print("\n⚠️  No OCRBench mirror worked. Try manually checking:")
    print("   https://huggingface.co/datasets?search=OCRBench")

# Fix for PyGILState_Release crash: flush stdout before Python tears down threads
import sys
atexit.register(sys.stdout.flush)