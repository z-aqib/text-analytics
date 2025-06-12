from datetime import datetime   

# Capture start time
s_start_time = datetime.now()

print("Last time code executed:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# === INSTALL REQUIRED LIBRARIES (ONLY IN NOTEBOOKS) ===
%pip install transformers datasets accelerate peft bitsandbytes trl evaluate nltk

# === IMPORTS ===
import torch
import time
import os
import csv
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# === DPO VARIABLES BLOCK ===

BASE_MODEL = "TinyLlama/TinyLlama_v1.1"
# Base pretrained model

LORA_MODEL_PATH = "/kaggle/input/tinyllama-lora-sft"  # ← UPDATE THIS TO WHERE YOU SAVED LoRA MODEL
# Path to trained LoRA adapter

DATASET_NAME = "Intel/orca_dpo_pairs"
# Human preference dataset (prompt, chosen, rejected)

DPO_BETA = 0.1
# Preference strength. Try 0.05, 0.1, 0.3

DPO_LEARNING_RATE = 5e-5
# DPO training learning rate. Try 1e-5 to 5e-5

DPO_BATCH_SIZE = 4
# Per-device batch size

DPO_EPOCHS = 3
# Number of epochs for DPO training

DPO_GRAD_ACCUM = 4
# Gradient accumulation steps

MAX_LENGTH = 512
# Max token length for prompts

DPO_OUTPUT_DIR = "./tinyllama-lora-dpo"
# Where to save the final DPO model

FP16 = True
USE_BF16 = False


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)


dpo_dataset = load_dataset(DATASET_NAME, split="train")

def format_for_dpo(example):
    return {
        "prompt": example["question"],
        "chosen": example["response_1"],
        "rejected": example["response_2"]
    }

dpo_dataset = dpo_dataset.map(format_for_dpo)
dpo_dataset = dpo_dataset.shuffle(seed=42).select(range(5000))  # or fewer if needed


dpo_config = DPOConfig(
    beta=DPO_BETA,
    learning_rate=DPO_LEARNING_RATE,
    per_device_train_batch_size=DPO_BATCH_SIZE,
    num_train_epochs=DPO_EPOCHS,
    gradient_accumulation_steps=DPO_GRAD_ACCUM,
    max_length=MAX_LENGTH,
    output_dir=DPO_OUTPUT_DIR,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    evaluation_strategy="no",
    fp16=FP16,
    bf16=USE_BF16
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer
)

start_time = datetime.now()
print("Start time:", start_time.strftime('%Y-%m-%d %H:%M:%S'))


trainer.train()

end_time = datetime.now()
duration = end_time - start_time
training_minutes = round(duration.total_seconds() / 60, 2)

model.save_pretrained(DPO_OUTPUT_DIR)
tokenizer.save_pretrained(DPO_OUTPUT_DIR)

eval_prompts = [
    "Explain the difference between machine learning and deep learning.",
    "What is the Pythagorean theorem used for?",
    "List three causes of World War II.",
    "Translate 'Good morning' into French.",
    "What are some benefits of daily exercise?",
    "Summarize the plot of Romeo and Juliet.",
    "What does HTTP stand for?",
    "Give an example of a palindrome.",
    "How do you boil an egg perfectly?",
    "Describe the lifecycle of a butterfly."
]

reference_answers = [
    "Machine learning is a broader concept of algorithms that learn from data. Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
    "The Pythagorean theorem helps calculate the length of a side in a right triangle: a² + b² = c².",
    "Three causes of WWII include the Treaty of Versailles, the rise of fascism, and the invasion of Poland by Nazi Germany.",
    "‘Good morning’ in French is ‘Bonjour’.",
    "Benefits of daily exercise include improved mood, better sleep, and stronger muscles.",
    "Romeo and Juliet is a tragedy about two young lovers from feuding families who ultimately die because of misunderstandings and conflict.",
    "HTTP stands for HyperText Transfer Protocol.",
    "A palindrome is a word like 'racecar' that reads the same backward and forward.",
    "Boil an egg by placing it in boiling water for 9-12 minutes depending on the desired hardness.",
    "A butterfly’s lifecycle includes egg, larva (caterpillar), pupa (chrysalis), and adult stages."
]

smoother = SmoothingFunction().method2

def generate_response(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).split("### Response:")[-1].strip()

def calculate_bleu_scores(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother)
        scores.append(score)
    return scores

model.eval()

# Run evaluation
dpo_outputs = [generate_response(p, model, tokenizer) for p in eval_prompts]
bleu_dpo_scores = calculate_bleu_scores(dpo_outputs, reference_answers)
bleu_dpo_avg = round(sum(bleu_dpo_scores) / len(bleu_dpo_scores), 4)

print("DPO BLEU Score:", bleu_dpo_avg)

# Format log
log = {
    "timestamp": start_time.strftime('%Y-%m-%d %H:%M:%S'),
    "training_time": round(duration.total_seconds() / 60, 2),
    "base_model": BASE_MODEL,
    "lora_model_path": LORA_MODEL_PATH,
    "dataset": DATASET_NAME,
    "dpo_beta": DPO_BETA,
    "dpo_learning_rate": DPO_LEARNING_RATE,
    "dpo_batch_size": DPO_BATCH_SIZE,
    "dpo_epochs": DPO_EPOCHS,
    "dpo_grad_accum": DPO_GRAD_ACCUM,
    "max_length": MAX_LENGTH,
    "bleu_dpo_avg": bleu_dpo_avg
}

for i in range(10):
    log[f"q{i+1}_output"] = dpo_outputs[i]
    log[f"bleu{i+1}"] = round(bleu_dpo_scores[i], 4)

# Save to CSV
csv_path = "/kaggle/working/dpo_experiments_log.csv"
file_exists = os.path.isfile(csv_path)

with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=log.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(log)
