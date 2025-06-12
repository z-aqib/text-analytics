# ITA Assignment 5: Fine-Tuning TinyLlama using LoRA and DPO

## Team Members

* **Zehra Ahmed** (26965)
* **Farah Inayat** (26912)
* **Zuha Aqib** (26106)

GitHub Repository: [Click here to explore our project](https://github.com/z-aqib/text-analytics/tree/main/A5_LLM_Fine_Tune)

## Objective

The goal of this assignment was to explore two fine-tuning strategies on a small Large Language Model (LLM), **TinyLlama-1.1B-Chat**, using:

1. **Supervised Fine-Tuning via LoRA (Low-Rank Adaptation)**
2. **Preference Fine-Tuning via DPO (Direct Preference Optimization)**

We aimed to enhance model performance using different hyperparameters and compare the improvements using BLEU scores and manual evaluation metrics like **Helpfulness**, **Harmlessness**, and **Instruction Relevance**.

## Platform & Tools

* **Platform**: Kaggle (Free GPU: T4 and A100)
* **Language**: Python 3.10
* **Key Libraries**:

  * `transformers==4.40.0`
  * `datasets==2.18.0`
  * `peft==0.10.0`
  * `accelerate==0.29.0`
  * `nltk`, `scikit-learn`

Install with:

```bash
pip install transformers datasets peft accelerate nltk scikit-learn
```

## Datasets Used

### 1. Supervised Fine-Tuning (LoRA)

* **Dataset**: [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned)
* **Samples Used**: 5,000
* **Why?** Balanced subset for GPU feasibility, diverse and instructional.

### 2. Preference Fine-Tuning (DPO)

* **Dataset**: [`Intel/orca_dpo_pairs`](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
* **Samples Used**: 5,000 (also tried 1k, 3k, and 10k)
* **Why?** Provides pairs of human-ranked responses for preference learning.

## Preprocessing Steps

* Tokenized inputs and outputs using **TinyLlama’s tokenizer**.
* Filtered to **≤ 512 tokens**.
* For DPO: reformatted dataset into `(prompt, chosen, rejected)` tuples.
* Ensured consistent truncation and padding across all inputs.

## Experimental Setup

### LoRA Fine-Tuning

We ran **29 experiments** with variations across:

* `num_samples`: \[1000, 3000, 5000, 10000]
* `lora_r`: \[4, 8, 16, 32]
* `lora_alpha`: \[16, 32, 64]
* `dropout`: \[0.01, 0.05, 0.1]
* `target_modules`: \["q\_proj", "v\_proj"], \["k\_proj", "o\_proj"], all combined
* `epochs`: \[1, 2, 3, 4]
* `learning_rate`: \[1e-4, 2e-4, 3e-4, 5e-4]
* `batch_size`: \[2, 4, 6, 8]
* `grad_accum`: \[2, 4, 5]

**Best LoRA Configuration:**

```yaml
- r: 8
- alpha: 16
- dropout: 0.05
- target_modules: ["q_proj", "v_proj"]
- learning_rate: 0.0002
- batch_size: 4
- epochs: 2
- samples: 5000
```

**BLEU Score (Best LoRA)**: 0.0422

### DPO Fine-Tuning

We ran **7 experiments** using the **best LoRA model** as base.

* `num_samples`: \[1000, 3000, 5000, 10000]
* `beta`: \[0.05, 0.1, 0.3]
* `learning_rate`: \[1e-5, 5e-5]
* `batch_size`: \[4, 8]
* `epochs`: \[3, 10]

**Best DPO Configuration:**

```yaml
- beta: 0.1
- learning_rate: 5e-5
- batch_size: 8
- epochs: 3
- samples: 3000
```

**BLEU Score (Best DPO)**: 0.0388 (Note: DPO scores were mainly evaluated **manually**)

## Evaluation Metrics

### BLEU Score

Used to evaluate overlap between generated and reference answers.

### Manual Evaluation Dimensions

* **Helpfulness**: Fulfills the instruction clearly.
* **Harmlessness**: Free from offensive or unsafe content.
* **Relevance**: Stays aligned with the prompt.

## Key Insights

### LoRA Observations

* Increasing `r` and `alpha` helped but plateaued at higher values.
* Best results came from moderate dropout (0.05) and balanced batch size (4).
* BLEU didn't always capture improvements like fluency or clarity.
* Smaller batch size with smart regularization gave better results than just scaling.

### DPO Observations

* Lower `beta` produced more helpful and safe answers.
* Higher `beta` improved focus but reduced answer richness.
* Smaller batch size helped fine-grained learning.
* Manual evaluation showed **DPO outperformed LoRA**, especially in instruction following.

## Reproducibility

We provide:

* CSV logs of every experiment
* Notebooks for each trial
* Top 5 LoRA and DPO experiments
* [Excel Summary Sheet](https://github.com/z-aqib/text-analytics/tree/main/A5_LLM_Fine_Tune)

To reproduce:

1. Run `LoRA` fine-tuning notebook with best config.
2. Load saved LoRA model and run DPO notebook.
3. Use `bleu_eval.py` or provided code to compute BLEU scores.

## Limitations & Notes

* **BLEU Limitations**: Did not capture qualitative gains (better wording, structure).
* **Training Limits**: Short training time and limited data = limited impact on BLEU.
* **Base Model Strength**: TinyLlama was already strong, so LoRA gains were subtle.
* **DPO Needed**: Human preference alignment (DPO) gave more visible improvements.

## Final Verdict

**Best Results = LoRA + DPO Combo!**
LoRA improved fluency and task adaptation. DPO further refined helpfulness, safety, and alignment—making it essential for instruction tuning.

This assignment taught us the art of **fine-tuning LLMs** under constraints, and how evaluation needs both metrics and manual inspection for meaningful conclusions.