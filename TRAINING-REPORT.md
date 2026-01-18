# LoRA Fine-tuning Training Report

## Overview
This report documents the fine-tuning process of the Qwen2.5-1.5B-Instruct model using Low-Rank Adaptation (LoRA) with 4-bit quantization (QLoRA).

## Model Information
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization using bitsandbytes
- **Maximum Sequence Length**: 512 tokens

## Dataset Configuration

### Data Sources
- **Training Data**: `/kaggle/input/instruct-dataset/train.jsonl`
- **Validation Data**: `/kaggle/input/instruct-dataset/val.jsonl`

### Data Format
The dataset follows an instruction-tuning format with three components:
- `instruction` - Task description
- `input` - Context or additional information
- `output` - Expected response

### Prompt Template
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

## Tokenization

### Tokenizer Configuration
- Uses the AutoTokenizer from the base model
- Truncation enabled at 512 tokens
- Labels created by copying input_ids for causal language modeling

## Model Configuration

### Quantization Settings (BitsAndBytes)
- **Precision**: 4-bit quantization
- **Quantization Type**: NF4 (Normal Float 4)
- **Compute Dtype**: float16

The model is loaded with 4-bit quantization to reduce memory footprint while maintaining performance. The `prepare_model_for_kbit_training` function prepares the quantized model for efficient training.

### LoRA Configuration
- **Rank (r)**: 16
- **LoRA Alpha**: 32
- **Dropout**: 0.05
- **Bias**: None
- **Task Type**: Causal Language Modeling
- **Target Modules**: 
  - `q_proj` (Query projection)
  - `v_proj` (Value projection)

The LoRA configuration applies low-rank adaptation to the attention mechanism's query and value projections, significantly reducing the number of trainable parameters while maintaining model quality.

## Training Configuration

### Hyperparameters
- **Training Epochs**: 3
- **Per-Device Train Batch Size**: 2
- **Per-Device Eval Batch Size**: 2
- **Gradient Accumulation Steps**: 2
- **Effective Batch Size**: 4 (2 × 2)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW (PyTorch implementation)
- **Mixed Precision**: FP16 enabled
- **Logging Steps**: Every 10 steps
- **Save Strategy**: After each epoch

### Data Collation
A custom `MinimalCausalLMCollator` is implemented to:
- Handle dynamic padding of input sequences
- Pad labels with -100 (ignored in loss calculation)
- Ensure consistent batch tensor dimensions

## Training Efficiency

### Parameter Efficiency
The LoRA approach dramatically reduces the number of trainable parameters. The `print_trainable_params` function reports:
- Total model parameters
- Trainable parameters (LoRA adapters only)
- Percentage of trainable parameters (typically <1% of total)

This makes fine-tuning feasible on consumer hardware while requiring significantly less memory and compute resources.

## Output and Artifacts

### Model Saving
- **Adapter Directory**: `./adapters`
- Saves LoRA adapter weights only (not the full model)
- Tokenizer configuration saved alongside adapters
- Compressed output: `adaptermodel.zip`

### Benefits of Adapter-Only Saving
- Minimal storage footprint (only LoRA weights, typically a few MB)
- Easy deployment by loading adapters on top of base model
- Facilitates version control and experimentation
- Enables multi-adapter scenarios

## Training Process Summary

1. **Environment Setup** - Install required libraries
2. **Data Loading** - Load instruction dataset from JSONL files
3. **Tokenization** - Convert text to tokens with proper formatting
4. **Model Loading** - Load base model with 4-bit quantization
5. **LoRA Application** - Apply LoRA adapters to target modules
6. **Training** - Fine-tune for 3 epochs with specified hyperparameters
7. **Save Adapters** - Export trained LoRA weights
8. **Compression** - Create ZIP archive of adapters
