# FastVLM CPU Benchmarks

This repository contains CPU benchmarking tools and evaluation scripts for FastVLM, focused on multimodal reasoning, OCR, and text-based visual question answering.

| Benchmark | Tested resolution | Accuracy metric | Value |
|---|---|---|---|
| POPE | 512×512 | Accuracy | 80.39% |
| GQA | 512×512 | Accuracy | 60.12% |
| TextVQA | 1024×1024 | VQA accuracy | 72.21% |
| OCRBench v2 | 1024×1024 | OCR accuracy | 45.57% |

## What this repo includes

- `fastvlm_bench.py` — main benchmark runner for core performance tests
- `bench_llava_wild.py` — benchmark scripts for LLAVA wild-style visual reasoning
- `bench_llava_wild_512.py` — lower-resolution 512px variant for faster CPU inference
- `bench_test_api.py` — API-level benchmark and latency testing
- `download_textVQA_OCRbench.py` — dataset download helper for TextVQA and OCRBench
- `explore_gqa.py` — experiment script for GQA dataset analysis
- `fastvlm_eval_pope.py` — evaluation script for POPE benchmark
- `fastvlm_ocrbench_eval.py` — OCRBench evaluation script
- `fastvlm_pope_eval_all.py` — full POPE benchmark runner
- `fastvlm_textvqa_eval.py` — TextVQA evaluation script
- `fastvlm_textvqa_eval_old.py` — legacy TextVQA evaluation flow
- `final_textvqa.py` — final TextVQA submission/evaluation wrapper
- `extract_images.py` — image extraction utility for dataset preprocessing

## Benchmark datasets and reports

This repo also stores benchmark results and checkpoint files used for evaluation:

- `gqa_checkpoint_partial.json`
- `gqa_final_benchmark_report.json`
- `ocrbench_benchmark_20260627_182825.json`
- `ocrbench_checkpoint_partial.json`
- `pope_benchmark_20260622_010526.json`
- `pope_checkpoint_partial.json`
- `textvqa_benchmark_20260624_235116.json`
- `textvqa_benchmark_20260628_005523_1024.json`
- `updated_file_textvqa.json`

## Benchmark results summary

The following table shows the key accuracy and latency metrics with the tested resolution for each benchmark.

| Benchmark | Tested resolution | Key metrics | Values |
|---|---|---|---|
| LLAVA Wild | 512×512 | Avg time-to-first-token | 1382.53 ms |
| LLAVA Wild | 512×512 | Min TTFT | 1204.12 ms |
| LLAVA Wild | 512×512 | Max TTFT | 2278.28 ms |
| LLAVA Wild | 512×512 | Avg total pipeline latency | 5975.24 ms |
| POPE | 512×512 | Accuracy | 0.8039 |
| POPE | 512×512 | Precision | 0.9331 |
| POPE | 512×512 | Recall | 0.6546 |
| POPE | 512×512 | F1 score | 0.7694 |
| GQA | 512×512 | Accuracy | 0.6012 |
| GQA | 512×512 | Precision | 0.5974 |
| GQA | 512×512 | Recall | 0.7111 |
| GQA | 512×512 | F1 score | 0.6493 |
| TextVQA | 1024×1024 | VQA accuracy | 0.7221 |
| TextVQA | 1024×1024 | Exact match rate | 0.7744 |
| OCRBench v2 | 1024×1024 | OCR accuracy | 0.4557 |

### Detailed notes
- `POPE` and `GQA` were evaluated with the 512×512 image encoder.
- `LLAVA Wild` latency results were measured on 512×512 samples.
- `TextVQA` and `OCRBench v2` were tested on 1024×1024 resolution.

## Purpose

This repository is designed to:

- measure FastVLM accuracy and runtime on standard visual reasoning benchmarks
- compare CPU inference performance for 1024×1024 and 512×512 vision encoder variants
- validate model behavior for OCR, TextVQA, GQA, and POPE tasks
- support reproducible benchmark results with stored report files

## How to use

1. Install dependencies for Python benchmarking, e.g.:

```bash
pip install numpy pillow onnxruntime requests
```

2. Download benchmark datasets if needed:

```bash
python download_textVQA_OCRbench.py
```

3. Run a benchmark script:

```bash
python fastvlm_bench.py
```

4. Run a specific evaluation:

```bash
python fastvlm_textvqa_eval.py
python fastvlm_ocrbench_eval.py
python fastvlm_eval_pope.py
```

5. Test API performance:

```bash
python bench_test_api.py
```

## Notes

- The repository includes both standard and legacy evaluation flows.
- `bench_llava_wild.py` and `bench_llava_wild_512.py` are useful when comparing 1024 and 512 resolution CPU inference tradeoffs.
- Benchmark reports are provided for reference and can be used to reproduce results.

## Recommended workflow

- Use `download_textVQA_OCRbench.py` first to fetch datasets.
- Run the task-specific script for the benchmark you want to evaluate.
- Compare output JSON reports with stored checkpoint files for consistency.
- Use low-resolution (`_512`) benchmarks when optimizing for CPU latency.

## License

This repo follows the same licensing terms as the broader FastVLM project. See the main project `LICENSE` file for details.
