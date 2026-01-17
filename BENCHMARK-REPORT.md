# LLM Benchmark Report

## Overview

This report documents the end-to-end benchmarking setup used to evaluate locally served Large Language Models (LLMs). The goal was to measure **latency**, **throughput**, and **semantic accuracy** while running models entirely on local infrastructure.

The benchmarking pipeline covers:

* Serving models locally (vLLM and llama.cpp)
* Running parallel, streaming inference
* Capturing performance metrics
* Computing semantic accuracy using embeddings

---

## Model Serving Setup

### 1. vLLM (Transformer Models)

Transformer-based models were served locally using **vLLM**, which provides:

* OpenAI-compatible `/v1/chat/completions` API
* Continuous batching
* Efficient KV-cache management
* Streaming token support

1. Install the vllm for the cpu using the given docs link:
    https://docs.vllm.ai/en/stable/getting_started/installation/cpu/

2. server the model locally use the command:
```bash
    vllm serve <model_path> --enforce-eager --enable-prompt-tokens-details --port 8080
```

---

### 2. GGUF Model via llama.cpp

Quantized GGUF models were served using **llama.cpp** in server mode.

Key characteristics:

* Supports low-bit quantized models (Q4, Q5, Q8)
* Exposed via an OpenAI-compatible HTTP endpoint

1. Install llama.cpp
Clone the repository from GitHub:
```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
```
2. Build the project
```bash
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
```

3. Serve the model using the command:
```bash
    llama.cpp/build/bin/llama-server --model src/quantized/model-q4_0.gguf --port 8080 --ctx-size 2048
```
---

## Benchmarking Pipeline (`test_inference.py`)

The benchmarking script performs **continuous batch streaming inference** with the following flow:

### 1. Input Data

Prompts are loaded from a JSON file with the structure:

```json
{
  "prompt": "<question>",
  "output": "<expected_answer>"
}
```

The expected output is used only for accuracy evaluation.

---

### 2. Batch Execution

* Prompts are split into fixed-size batches
* Each batch is executed sequentially
* Prompts within a batch are executed **in parallel** using `asyncio.gather`

---

### 3. Streaming Inference

For each prompt:

* A streaming request is sent to the local LLM server
* Tokens are received incrementally
* Tokens are printed live as they arrive

---

## Metrics Collected

For every prompt, the following metrics are computed:

### 1. Latency

* Total wall-clock time from request start to final token

### 2. TTFT (Time To First Token)

* Time between request submission and first generated token

### 3. Tokens per Second

* Calculated as: total_tokens / total_latency


### 4. Semantic Accuracy

Accuracy is measured using **semantic similarity**, not exact string matching.

Process:

1. Generated answer and expected answer are embedded using `sentence-transformers/all-MiniLM-L6-v2`
2. Embeddings are normalized
3. FAISS Inner Product index is used
4. Cosine similarity score is returned as accuracy (0–1)

This approach captures meaning-level correctness rather than surface text overlap.

---

## Results Storage

All benchmark results are appended to:

```
src/benchmarks/results.csv
```

Each row contains:

* Model identifier
* Prompt ID
* Prompt text
* Token count
* Latency (seconds)
* TTFT (seconds)
* Tokens per second
* Semantic accuracy score

---
