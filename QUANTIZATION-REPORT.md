# Model Quantization Report

## Overview
This report documents the quantization pipeline that converts the fine-tuned LoRA model into multiple quantized formats for efficient deployment across different hardware configurations.

## Process Summary
The quantization pipeline performs the following steps:
1. Merge LoRA adapters with base model
2. Save merged model in FP16 precision
3. Create INT8 quantized version
4. Create INT4 quantized version
5. Convert to GGUF format for llama.cpp compatibility
6. Apply Q4_0 quantization for maximum compression

## Base Configuration

### Model Information
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Adapter Source**: `/kaggle/input/adapters` (LoRA weights from training)

### Output Structure
```
./quantized/
├── merged-fp16/     # Full precision merged model
├── model-int8/      # 8-bit quantized model
├── model-int4/      # 4-bit quantized model
├── model.gguf       # GGUF format (FP16)
└── model-q4_0.gguf  # GGUF format (Q4_0 quantized)
```

## Quantization Stages

### Stage 1: Merge LoRA Adapters with Base Model

**Purpose**: Combine the trained LoRA adapter weights back into the base model

**Process**:
- Load base model in FP16 precision
- Load LoRA adapters using PeftModel
- Merge adapters into base model using `merge_and_unload()`
- Save complete merged model

**Output**: 
- Directory: `./quantized/merged-fp16/`
- Precision: FP16 (16-bit floating point)
- Contents: Full model weights + tokenizer configuration

**Characteristics**:
- Highest quality and precision
- Largest file size
- Suitable for GPU inference with ample VRAM
- Serves as source for further quantization

### Stage 2: INT8 Quantization

**Purpose**: Reduce model size by ~50% with minimal quality degradation

**Configuration**:
```python
BitsAndBytesConfig(load_in_8bit=True)
```

**Process**:
- Load merged FP16 model
- Apply 8-bit quantization using bitsandbytes library
- Save quantized weights

**Output**:
- Directory: `./quantized/model-int8/`
- Precision: INT8 (8-bit integer)
- Size: ~50% of FP16 model

**Characteristics**:
- Balanced quality-size tradeoff
- Suitable for mid-range GPUs
- Minimal performance degradation
- Good for production deployments with moderate memory constraints

### Stage 3: INT4 Quantization (NF4)

**Purpose**: Maximum size reduction with acceptable quality loss

**Configuration**:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**Key Settings**:
- **Quantization Type**: NF4 (Normal Float 4) - optimized 4-bit format
- **Compute Dtype**: FP16 - computations performed in half precision
- **Double Quantization**: Enabled - quantizes quantization constants for additional compression

**Process**:
- Load merged FP16 model
- Apply NF4 4-bit quantization
- Enable double quantization for maximum compression
- Save quantized weights

**Output**:
- Directory: `./quantized/model-int4/`
- Precision: NF4 (4-bit normal float)
- Size: ~25% of FP16 model

**Characteristics**:
- Maximum compression
- Suitable for consumer GPUs and edge devices
- Acceptable quality degradation for most tasks
- Ideal for resource-constrained deployments

### Stage 4: GGUF Conversion (llama.cpp Format)

**Purpose**: Enable CPU and heterogeneous inference using llama.cpp

**Process**:
1. Clone llama.cpp repository
2. Install Python requirements
3. Convert merged FP16 model to GGUF format using `convert_hf_to_gguf.py`
4. Build llama.cpp with CMake
5. Apply Q4_0 quantization using llama.cpp's quantization tool

**GGUF Format Benefits**:
- CPU-optimized inference
- Cross-platform compatibility
- Efficient memory mapping
- Support for various quantization schemes
- No Python runtime dependency

**Q4_0 Quantization**:
```bash
llama-quantize model.gguf model-q4_0.gguf q4_0
```

**Output**:
- `model.gguf` - GGUF format in FP16
- `model-q4_0.gguf` - GGUF with Q4_0 quantization

**Q4_0 Characteristics**:
- 4-bit quantization optimized for llama.cpp
- Fastest CPU inference
- Smallest file size among GGUF variants
- Suitable for CPU-only deployments

## Model Size Comparison

The pipeline includes a size comparison utility that reports disk usage for each quantization level:

**Expected Size Reduction** (approximate):
- **FP16 (Baseline)**: 100% (~3GB for 1.5B model)
- **INT8**: ~50% (~1.5GB)
- **INT4/NF4**: ~25% (~750MB)
- **Q4_0 GGUF**: ~25% (~750MB)

## Quantization Formats Summary

| Format | Precision | Library | Use Case | Size | Quality |
|--------|-----------|---------|----------|------|---------|
| FP16 | 16-bit float | Transformers | High-end GPU | Largest | Best |
| INT8 | 8-bit integer | BitsAndBytes | Mid-range GPU | 50% | Excellent |
| INT4/NF4 | 4-bit NF | BitsAndBytes | Consumer GPU | 25% | Good |
| Q4_0 GGUF | 4-bit | llama.cpp | CPU/Edge | 25% | Good |
