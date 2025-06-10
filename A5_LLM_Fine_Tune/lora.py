#!/usr/bin/env python
# coding: utf-8

# Acknowledgements
# 
# Name: Zuha Aqib, Farah Inayat, Zehra Ahmed   
# Date: 8th June 2025   
# ITA Assignment 5

# In[1]:


from datetime import datetime   

# Capture start time
s_start_time = datetime.now()

print("Last time code executed:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# # Pre Reqs

# ## Installing Packages
# here we install the necessary packages

# In[2]:


get_ipython().run_line_magic('pip', 'install transformers datasets accelerate peft bitsandbytes trl evaluate nltk')


# ## Imports

# In[3]:


import torch
print("GPU is available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# In[4]:


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import torch
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import DataCollatorForLanguageModeling


from datetime import datetime
import csv
import os


# # Variables
# here we set the changeable variables that we will change during the course of experimenting - all other variables will not be changed

# In[5]:


# === MODEL & DATASET CONFIGURATION ===

BASE_MODEL = "TinyLlama/TinyLlama_v1.1"  
# TinyLlama base model (1.1B parameters, instruction-tuned)

DATASET_NAME = "yahma/alpaca-cleaned"
# Dataset with instruction–response pairs for general fine-tuning

NUM_SAMPLES = 10000  
# Total number of examples to use from the dataset.
# Range: 1000–5000 (lower = faster training, higher = better quality)

# === LoRA CONFIGURATION ===

LORA_R = 32  
# LoRA rank — controls the number of trainable parameters added.
# Try: 4, 8, 16, 32

LORA_ALPHA = 32  
# LoRA scaling factor — higher values amplify changes made by low-rank layers.
# Try: 16, 32, 64

LORA_DROPOUT = 0.05  
# Dropout rate for regularization in LoRA layers.
# Try: 0.01, 0.05, 0.1

LORA_TARGET_MODULES = ["q_proj", "v_proj"]
# The attention submodules where LoRA is applied.
# Try: ["q_proj", "v_proj"], ["k_proj", "o_proj"], or all four.

# === TRAINING HYPERPARAMETERS ===

EPOCHS = 2  
# Number of full passes over the training data.
# Try: 1 for testing, 2–4 for actual fine-tuning

LEARNING_RATE = 2e-4  
# How fast the model learns.
# Try: 1e-4, 2e-4, 3e-4, 5e-4

BATCH_SIZE = 4  
# Number of examples processed per device per step.
# Depends on your GPU (2–8 is typical for Colab)

GRADIENT_ACCUMULATION = 4  
# To simulate larger batch size by accumulating gradients.
# Effective batch size = BATCH_SIZE × GRADIENT_ACCUMULATION

MAX_LENGTH = 512  
# Maximum token length per input text (longer = more context, but slower).
# Keep 512 for most LLMs unless your dataset has longer instructions.

FP16 = True  
# Use 16-bit floating point (mixed precision) for faster training and less memory.
# Keep True unless using BF16.

USE_BF16 = False  
# Use BF16 precision if you're on an A100 or TPU v4 (Google Colab Pro+ or Kaggle High-RAM).
# Only set True if you know your hardware supports it.

# === OUTPUT CONFIGURATION ===

OUTPUT_DIR = "/kaggle/working/tinyllama-lora-sft"
# Folder where the fine-tuned model and checkpoints will be saved.
# You can rename this for each experiment (e.g. "./trial1", "./lora_r16_epoch3", etc.)


# # Load the model
# here we loadthe tiny llama model and tokenizer

# ## Login Hugging Face

# In[6]:


from huggingface_hub import login

login("")


# ## Load the model

# In[7]:


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": 0},  # Force all model to cuda:0
    load_in_8bit=True
)

print("Model and tokenizer loaded successfully.")


# In[8]:


# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)


# # Load Dataset
# here we load the dataset and fix its printing

# ## Load the dataset

# In[9]:


# Load the Alpaca-cleaned dataset
dataset = load_dataset(DATASET_NAME)

print(dataset["train"][0])  # Show a sample


# In[10]:


# Print the number of examples in the 'train' split
print("Total number of samples in dataset:", len(dataset["train"]))


# ## Preprocess into Prompt-Response Format

# In[11]:


def format_alpaca(example):
    instruction = example["instruction"]
    input_text = example["input"]
    response = example["output"]

    # Combine with proper formatting
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += f"\n### Response:\n{response}"

    return {"text": prompt}

# Format the full dataset
formatted_dataset = dataset["train"].map(format_alpaca)


# ## Limit to a certain amount of records
# because training is taking too long lets limit to 3000 records for now

# In[12]:


small_dataset = formatted_dataset.shuffle(seed=42).select(range(NUM_SAMPLES))


# In[13]:


print(small_dataset[0]["text"])


# ## Tokenize the dataset

# In[14]:


# Tokenize and add labels for causal language modeling (no masking)
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = small_dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[15]:


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Important: this is a causal LM, not masked
)


# # Lora

# ## Configure lora

# In[16]:


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


# ## Training arguments

# In[17]:


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=FP16,
    bf16=USE_BF16,
    report_to="none"
)


# In[18]:


trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    peft_config=lora_config,
    data_collator=data_collator
)


# ## Start Training

# In[19]:


print(next(model.parameters()).device)  # model device
print(type(tokenized_dataset[0]['input_ids']))

print(tokenized_dataset[0]['input_ids'].device)  # should show cpu


# In[20]:


sample_input = tokenized_dataset[0]['input_ids']
print(type(sample_input))  # likely list

sample_tensor = torch.tensor(sample_input)
print(sample_tensor.device)  # this will be cpu by default


# In[21]:


trainer.model = trainer.model.to("cuda")


# In[22]:


# === START TIMER BEFORE TRAINING ===
start_time = time.time()


# In[23]:


trainer.train()


# In[24]:


# === END TIMER AFTER TRAINING ===
end_time = time.time()
training_minutes = round((end_time - start_time) / 60, 2)


# In[25]:


# After training
trainer.model.save_pretrained("tinyllama-lora-sft")     # <- This saves LoRA adapter correctly
tokenizer.save_pretrained("tinyllama-lora-sft")


# # Evaluation
# here we evaluate with 10 evaluation prompts and BLEU

# ## Eval prompts with answers

# In[26]:


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


# In[27]:


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


# ## Test model
# first we run the prompts on our `BASE_MODEL` i.e. the model we started with and then we run the prompts on our `FINAL_MODEL` i.e. the model we have fine-tuned.

# In[28]:


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


# In[29]:


from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", load_in_8bit=True)
lora_model = PeftModel.from_pretrained(base_model, "/kaggle/working/tinyllama-lora-sft")

base_outputs = [generate_response(p, base_model, tokenizer) for p in eval_prompts]
lora_outputs = [generate_response(p, lora_model, tokenizer) for p in eval_prompts]


# ## Compute BLEU

# In[30]:


smoother = SmoothingFunction().method2

def calculate_bleu_scores(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        scores.append(score)
    return scores


# In[31]:


bleu_base_scores = calculate_bleu_scores(base_outputs, reference_answers)
bleu_lora_scores = calculate_bleu_scores(lora_outputs, reference_answers)

bleu_base_avg = round(sum(bleu_base_scores) / len(bleu_base_scores), 4)
bleu_lora_avg = round(sum(bleu_lora_scores) / len(bleu_lora_scores), 4)

print(f"Base Model BLEU Avg: {bleu_base_avg:.4f}")
print(f"LoRA Model BLEU Avg: {bleu_lora_avg:.4f}")


# # Save configuration
# here we save all the run in an existing or new `csv` file

# ## First comment on the results

# In[32]:


# === ANALYSIS AUTO-GENERATOR ===
improved = sum(l > b for b, l in zip(bleu_base_scores, bleu_lora_scores))
same = sum(l == b for b, l in zip(bleu_base_scores, bleu_lora_scores))
worsened = sum(l < b for b, l in zip(bleu_base_scores, bleu_lora_scores))

comment_parts = []

# Overall BLEU delta
delta = bleu_lora_avg - bleu_base_avg
if delta > 0.1:
    comment_parts.append("Significant BLEU improvement.")
elif delta > 0.03:
    comment_parts.append("Moderate BLEU improvement.")
elif delta > 0:
    comment_parts.append("Slight BLEU gain.")
elif delta < -0.03:
    comment_parts.append("BLEU score dropped.")
else:
    comment_parts.append("No major BLEU change.")

# Per-question analysis
comment_parts.append(f"Improved in {improved}/10 prompts, same in {same}, worse in {worsened}.")

# Time efficiency
if training_minutes < 10:
    comment_parts.append("Trained quickly (<10 min).")
elif training_minutes < 20:
    comment_parts.append("Training time was reasonable.")
else:
    comment_parts.append("Training took longer than usual.")

# Final comment string
your_manual_comment = " ".join(comment_parts)
print("Auto-generated comment:", your_manual_comment)


# ## create csv and write to it

# In[33]:


csv_path = "/kaggle/working/tinyllama_experiments_log_2.csv"
file_exists = os.path.isfile(csv_path)


# In[34]:


experiment_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_time_mins": training_minutes,
    "base_model": BASE_MODEL,
    "dataset": DATASET_NAME,
    "num_samples": NUM_SAMPLES,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "lora_target_modules": str(LORA_TARGET_MODULES),
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRADIENT_ACCUMULATION,
    "max_length": MAX_LENGTH,
    "bleu_base_avg": bleu_base_avg,
    "bleu_lora_avg": bleu_lora_avg,
    "analysis": your_manual_comment
}


# In[35]:


# Add Q&A
for i in range(10):
    experiment_data[f"q{i+1}_base"] = base_outputs[i]
    experiment_data[f"q{i+1}_lora"] = lora_outputs[i]
    experiment_data[f"bleu{i+1}_base"] = bleu_base_scores[i]
    experiment_data[f"bleu{i+1}_lora"] = bleu_lora_scores[i]


# In[36]:


# Write to CSV
with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=experiment_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(experiment_data)


# # Last Executed Time

# In[37]:


# Capture end time
s_end_time = datetime.now()

# Compute time difference
diff = s_end_time - s_start_time

# Total seconds (float)
total_seconds = diff.total_seconds()

# Decompose
hours, rem = divmod(total_seconds, 3600)
minutes, rem = divmod(rem, 60)
seconds = int(rem)
milliseconds = diff.microseconds // 1000

# Display
print(f"Start time : {s_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"End time   : {s_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Duration   : {int(hours)}h {int(minutes)}m {seconds}s {milliseconds}ms")


# In[ ]:




