# Project Report: LLM Evaluation on Austrian Law

## Models Used

Two models are evaluated on 643 Austrian law questions from `dataset_clean.csv`. Both models receive the same prompts and produce German-language.

### Model 1: Gemma 4 31B (API Inference, Zero-Shot)

This model accessed through Google AI Studio API, the model relies entirely on its pre-training knowledge and a system prompt that instructs it to act as an Austrian tax law expert.

- **Model ID:** `gemma-4-31b-it` (instruction-tuned variant)
- **Parameters:** ~31 billion
- **Sampling:** Temperature = 0.1 (low, for factual accuracy)
- **System prompt:** "You are a highly qualified expert in Austrian tax law. Provide precise, professional answers. Always reference relevant paragraphs (e.g., EStG, KStG, BAO, UStG) where applicable. The answer must be in German and formatted as one continuous paragraph."
- **Inference setup:** Sequential API calls with 5-second delay per request (free-tier rate limit), retry logic for 429/503 errors via `tenacity`.

### Model 2: Gemma 4 E2B (QLoRA Fine-Tuned)

This is a smaller model fine-tuned on domain-specific German law data using QLoRA (Quantized Low-Rank Adaptation). The base model is loaded in 4-bit precision, and only small LoRA adapter layers are trained while the rest of the model stays frozen.

- **Model ID:** `google/gemma-4-E2B-it` (instruction-tuned variant)
- **Total parameters:** 2.3B effective (5.1B with embeddings)
- **Trainable parameters:** 4,644,864 (0.09% of total) — only the LoRA adapters
- **Sampling:** Temperature = 0.1
- **Fine-tuning data:** `DomainLLM/german-law-qa` from Hugging Face
- **Inference setup:** Local GPU inference using Unsloth's optimized 2x faster inference mode.

### Fine-Tuning Dataset
- **Dataset:** `DomainLLM/german-law-qa` from Hugging Face
- **Size:** 14,160 training examples of German law question-answer pairs
- **Format:** Each example formatted into Gemma's chat template: `<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn><eos>`
- **Domain:** General German law (not exclusively Austrian tax law)

### Fine-Tuning Method: QLoRA
QLoRA combines quantization with Low-Rank Adaptation to enable fine-tuning large models on limited hardware (Google Colab T4 GPU with 16 GB VRAM):

1. **Quantization:** Base model loaded in 4-bit precision (reduces memory from ~20 GB to ~5 GB)
2. **Freezing:** All base model weights are frozen (not updated during training)
3. **LoRA adapters:** Small 16-bit adapter matrices injected into attention layers, only these are trained

### Hyper-Parameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 8 |
| LoRA alpha | 8 |
| LoRA dropout | 0 (optimized by Unsloth) |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Batch size (per device) | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Training steps | 150 |
| Learning rate | 2e-4 |
| Warmup steps | 5 |
| LR scheduler | Linear |
| Optimizer | paged_adamw_8bit |
| Weight decay | 0.01 |
| Max sequence length | 1024 |
| Precision | FP16 (T4 GPU) |

## Evaluation

### Evaluation Approach

Each model's generated answers are compared against 643 human-written reference answers from `dataset_answer.csv`. The evaluation script is in `code/evaluation.ipynb`.

**Metrics used:**
- **ROUGE-1 / ROUGE-2 / ROUGE-L** (F1): Measures n-gram and subsequence overlap between prediction and reference. ROUGE-1 counts unigram matches, ROUGE-2 counts bigram matches, ROUGE-L uses the longest common subsequence.
- **BLEU**: Precision-based metric with brevity penalty. Uses smoothing (Method 1) since many answers are short and higher-order n-gram matches can be zero.
- **BERTScore** (F1): Uses contextual embeddings from `bert-base-multilingual-cased` to measure semantic similarity. More robust to paraphrasing than surface-level metrics.

### Main Results Table

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 |
|---|---|---|---|---|---|
| Gemma 4 31B (Inference) | 0.1816 | 0.0571 | 0.1152 | 0.0208 | 0.6891 |
| **Gemma 4 E2B (QLoRA Fine-tuned)** | **0.2167** | **0.0643** | **0.1572** | **0.0307** | **0.7113** |

### Which Model Performs Best?

**Model 2 (Gemma 4 E2B, QLoRA fine-tuned) outperforms Model 1 (Gemma 4 31B, zero-shot) across all metrics.** This is a notable result because Model 2 is smaller (5.1B vs 31B parameters). The fine-tuning on German law data gives it an edge even though the training data was general German law, not specifically Austrian law.

The performance gap is most visible in ROUGE-L (+0.042) and BLEU (+0.010), suggesting that fine-tuning helps the model produce answers that are structurally closer to the expected format.

## Error Analysis

### Score Distribution (ROUGE-L)

| Score Range | Model 1 (31B) | Model 2 (E2B-FT) |
|---|---|---|
| Zero (0.0) | 0 (0.0%) | 16 (2.5%) |
| Low (0.01 - 0.10) | 290 (45.1%) | 127 (19.8%) |
| Medium (0.10 - 0.30) | 349 (54.3%) | 464 (72.2%) |
| High (> 0.30) | 4 (0.6%) | 36 (5.6%) |

Model 2 has more answers in the medium and high range, but also has 16 answers with zero overlap (complete misses). Model 1 never scores exactly zero but clusters heavily in the low range.

### Main Issues Identified

**1. Excessive length (Model 1)**
Model 1 generates answers that are on average 1,310 characters long, it is about 4.7x the reference length (279 characters). This extreme verbosity dilutes precision: the model includes many correct concepts but keeps them in lengthy explanations that don't match the concise reference format.

**2. Hallucinated legal references (both models)**
Both models cite incorrect or irrelevant legal sources:

| Reference | Model 1 | Model 2 | Expected |
|---|---|---|---|
| KStG (tax law) | 118 | 287 | Correct for tax questions |
| EStG (income tax) | 445 | 177 | Correct for income tax questions |
| BGB (civil code) | 10 | 74 | Wrong — not relevant for tax law |

Model 2 hallucinates BGB (Buergerliches Gesetzbuch) references 74 times — likely because the fine-tuning dataset (`DomainLLM/german-law-qa`) covers general German law including civil law, causing confusion between legal domains.

**3. Repetitive generation (Model 2)**
About 3.9% of Model 2's answers contain repeated sentences (the model gets stuck in a loop and repeats the same phrase). This is a known issue with smaller models that haven't been trained long enough. Model 1 shows no repetition, likely due to its larger capacity and the API's generation safeguards.

**4. Wrong paragraph numbers (both models)**
Even when models identify the correct legal concept, they frequently cite wrong paragraph numbers. For example, Model 2 cites "§ 7 Abs. 1 KStG" for almost every question regardless of the actual relevant provision.

### Are Mistakes the Same Across Models?

No — each model has a distinct failure profile:

- **Model 1** tends to be **correct in substance but too long**. It identifies the right legal concepts but wraps them in lengthy explanations with excessive EStG references, resulting in low precision scores.
- **Model 2** tends to be **concise but sometimes factually wrong**. It matches the expected answer length well, but when it fails, it fails harder — citing completely wrong legal codes (BGB instead of KStG), repeating itself, or giving zero-overlap answers on topics outside its fine-tuning distribution.

## File Structure

```
Daniil_Rabotnev/
├── README.md                              # This project report
├── code/
│   ├── model1_inference.ipynb             # Model 1: Gemma 4 31B API inference
│   ├── model2_finetune.ipynb              # Model 2: Gemma 4 E2B QLoRA fine-tuning + inference
│   └── evaluation.ipynb                   # Evaluation script (ROUGE, BLEU, BERTScore)
└── results/
    ├── model1_api_inference_results.csv   # Model 1 predictions (643 answers)
    ├── model2_finetune_results.csv        # Model 2 predictions (643 answers)
    └── dataset_answers.csv                   # Gold-standard reference answers (645 entries)
```