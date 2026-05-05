# TurboQuant Experiment Pipeline

A comprehensive, self-contained evaluation suite for testing Google's **TurboQuant (ICLR 2026)** KV cache compression algorithm on real-world generative AI workloads. 

This repository provides the tooling to move beyond mathematical theory and measure how extreme low-bit KV cache quantization impacts Large Language Models (LLMs) across memory scaling, inference throughput, and downstream generation accuracy.

## ⚠️ Important Implementation Note
This script evaluates a **live runtime KV cache quantization prototype**. It quantizes cached keys and values dynamically between decoding steps. 

However, because native 3-bit tensor packing is not yet supported in standard PyTorch without custom kernels, **this prototype stores the quantized 3-bit values inside `int8` containers alongside per-vector scales**. 
* **Quality/Accuracy:** Accurately reflects true 3-bit TurboQuant behavior.
* **Memory/Throughput:** Reflects a prototype overhead. It achieves a ~1.98x memory reduction (instead of the theoretical 5.3x) and relies on Python-level dequantization during the forward pass. For production speedups, this algorithm must be paired with fused Triton/CUDA attention kernels.

## Supported Models
The pipeline defaults to testing the following models:
* `google/gemma-2b` (Base)
* `google/gemma-2b-it` (Instruct)
* `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

*Note: The external evaluator used for semantic similarity scoring is `sentence-transformers/all-MiniLM-L6-v2`.*

## The Evaluation Suite (E1 - E10)

The script automatically executes a 10-stage evaluation pipeline:

### Phase 1: Algorithm Characterization (Synthetic Tensors)
* **E1 - Bit-Depth Tradeoff:** Compares PolarQuant vs. TurboQuant (PolarQuant + QJL) on Attention MSE and KL Divergence at 2, 3, and 4 bits.
* **E2 - Distortion Rate Curve:** Plots empirical MSE against the theoretical $4^{-b}$ lower bounds.
* **E9 - Attention Entropy Analysis:** Measures whether quantization noise artificially flattens the model's attention distribution.
* **E4 - Memory Scaling:** Calculates theoretical (ideal 3-bit) vs. practical (int8+scale prototype) KV cache memory footprints from 512 up to 32,768 tokens.

### Phase 2: Live Model Evaluation
* **E3 - Layer Sensitivity:** Identifies which transformer layers are most vulnerable to quantization noise.
* **E10 - Mixed-Bit Precision Strategy:** Evaluates a sensitivity-ranked hybrid schedule (e.g., 4-bit for sensitive middle layers, 2-bit for robust late layers) to improve MSE while maintaining a ~2.94 effective bit rate.
* **E6 - Factual QA:** Tests zero-shot factual retrieval accuracy and semantic similarity against a baseline FP16 cache (20 questions × 3 seeds).
* **E8 - RAG Simulation:** Injects a dense context paragraph into the KV cache and tests if the model can still accurately extract facts at 3 bits.
* **E7 - Multi-Task Evaluation:** Measures semantic degradation across Factual, Reasoning, Coding, and Summarization prompts.
* **E5 - Latency/Throughput Sweep:** A robust benchmark sweeping prompt lengths (64, 512, 1024 tokens), capturing median tokens-per-second (TPS), Time-To-First-Token (TTFT), and latency variances.

## Installation

Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install torch transformers tqdm matplotlib numpy sentence-transformers
Note: A CUDA-enabled GPU is highly recommended. The script will automatically detect and utilize available GPUs.

Usage
Run the pipeline directly from the command line:

Bash
python turboquant_experiments.py
Resuming Incomplete Runs
The script features a robust state-saving mechanism. If the run is interrupted (e.g., due to an OOM error or manual cancellation), simply rerun the script. It will automatically detect the progress.json file in the latest output directory and resume exactly where it left off.

Outputs
All artifacts are saved to a timestamped directory: results/gemma2b_turboquant_<timestamp>/

experiment.log: Detailed runtime logs and metrics.

config.json: The configuration state for the run.

results.json: The complete, raw numerical data across all 10 experiments.

progress.json: State tracker for resumable runs.

plots/: A directory containing generated visualizations (e.g., Radar charts for multi-task similarity, bar charts for RAG accuracy, memory scaling plots, and a master SUMMARY_dashboard.png).

Modifying the Configuration
To test different context lengths, target tokens, or bit-depths, you can modify the Config class directly inside turboquant_experiments.py:

Python
class Config:
    model_id = "google/gemma-2b"
    bits_list = [2, 3, 4]
    default_bits = 3
    quantized_bits = 3
    # ... customize parameters here
Authors & Citation
Based on the TurboQuant algorithm by Zandieh, Daliri, Hadian, and Mirrokni (ICLR 2026).
If you use this evaluation suite for your own research or deployments, please consider linking back to this repository and reading the companion article on Towards Data Science.
