# Dataset Analysis & Preparation

## 1. Overview
This document describes the analysis and preparation of the dataset used for instruction tuning a Qwen-based LLM.

Key focuses:
- Instruction fine-tuning
- Quantization
- Optimized inference

Objective: construct a high-quality instruction-tuning dataset from raw, structured QA data (not to use an already instruction-ready dataset).

## 2. Raw dataset description
- Domain: Go programming language  
- Task type: Question–Answering (technical, code-heavy)  
- Source: structured StackOverflow-style QA dataset  
- Format: JSON

Raw sample structure:
```json
{
    "question_id": 1713214,
    "answer_id": 1721230,
    "title": "How to use C++ in Go",
    "question": "...",
    "answer": "..."
}
```

This dataset contains:
- Real-world developer questions
- Long-form answers with explanations and code snippets
- Minimal synthetic content

## 3. Subsampling
- Original size: ~35,000 samples  
- Selected subset for initial fine-tuning: 5,000 samples  

Rationale: for instruction alignment and domain adaptation, data quality and diversity matter more than sheer volume.

## 4. Instruction dataset construction
Since the raw dataset was not instruction-formatted, a custom instruction schema was applied.

Instruction template:
> You are an expert Go language developer.  
> Answer the following question clearly and accurately.

Final instruction-tuning format:
```json
{
    "instruction": "...",
    "input": "Title + Question",
    "output": "Answer"
}
```

Benefits:
- Preserves original QA semantics
- Avoids inheriting alignment bias from instruction-ready datasets
- Produces consistent and stable prompts for fine-tuning

## 5. Outlier removal
To improve training stability, token-based filtering was applied.

Criteria:
- Minimum tokens: 50
- Maximum tokens: 1200

Removed samples:
- Extremely short QA pairs (low information)
- Extremely long samples (risk of truncation and instability)

## 6. Dataset distribution
The final cleaned dataset exhibits:
- Strong dominance of technical explanations
- High code-to-text ratio
- Natural reasoning patterns

Distribution plots (saved):
```
src/data/stats/
├── token_hist.png
└── qa_length_dist.png
```

Plots generated for:
- Total token length
- QA token length

## 7. Train / validation split
- Split after cleaning and outlier removal to preserve token distribution parity  
- Train: 90%  
- Validation: 10%

## 8. Final dataset summary

| Property         | Value                        |
|------------------|------------------------------|
| Domain           | Go Programming               |
| Task Type        | QA (instruction-constructed) |
| Initial Samples  | 5,000                        |
| Final Samples    | 4,700-5,000                  |
| Train Split      | 90%                          |
| Validation Split | 10%                          |
| Format           | JSONL                        |
| Tokenizer        | Qwen                         |

## 9. Conclusion
This preparation process emphasizes control, transparency, and reproducibility. Constructing instruction data from raw QA pairs keeps fine-tuning interpretable and adaptable, suitable for studying model behavior under quantization and optimized inference settings.
