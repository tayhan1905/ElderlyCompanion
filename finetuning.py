import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    logging as hf_logging
)
from peft import LoraConfig, get_peft_model
import evaluate

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_info()

# -----------------------------
# Load dataset
# -----------------------------
logger.info("Loading Singlish→English dataset...")
ds = load_dataset("gabrielchua/singlish-to-english-synthetic")["train"].train_test_split(
    test_size=0.1, seed=42
)
train_ds, val_ds = ds["train"], ds["test"]
logger.info(f"Dataset loaded. Train size: {len(train_ds)} | Validation size: {len(val_ds)}")

def format_example(ex):
    return {
        "src": f"Translate Singlish to English:\nSinglish: {ex['singlish']}\nEnglish:",
        "tgt": ex["english"]
    }

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(format_example,   remove_columns=val_ds.column_names)
logger.info("Formatted dataset into instruction→target format.")

# -----------------------------
# Tokenizer
# -----------------------------
base_model = "google/mt5-base"
logger.info(f"Loading tokenizer for {base_model}...")
tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or "<pad>"
logger.info("Tokenizer loaded and pad token set.")

def tok_fn(batch):
    model_inputs = tok(batch["src"], max_length=128, truncation=True)
    with tok.as_target_tokenizer():
        labels = tok(batch["tgt"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logger.info("Tokenizing training and validation datasets...")
train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["src","tgt"])
val_tok   = val_ds.map(tok_fn,   batched=True, remove_columns=["src","tgt"])
logger.info("Tokenization complete.")

# -----------------------------
# Model + SALT (LoRA)
# -----------------------------
logger.info(f"Loading base model {base_model}...")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)

logger.info("Configuring SALT (LoRA) adapters...")
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q","k","v","o","wi","wo"],
    bias="none"
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# -----------------------------
# Training setup
# -----------------------------
logger.info("Preparing training setup...")
collator = DataCollatorForSeq2Seq(tok, model=model)

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = tok.batch_decode(preds, skip_special_tokens=True)
    labels = tok.batch_decode(labels, skip_special_tokens=True)
    bleu_res = bleu.compute(predictions=preds, references=[[l] for l in labels])
    chrf_res = chrf.compute(predictions=preds, references=labels)
    return {"bleu": bleu_res["score"], "chrf": chrf_res["score"]}

args = Seq2SeqTrainingArguments(
    output_dir="outputs-salt",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    bf16=torch.cuda.is_available(),
    report_to="none",
    logging_dir="logs"  # saves HF logs
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Training
# -----------------------------
logger.info("🚀 Starting training...")
trainer.train()
logger.info("✅ Training complete.")

# -----------------------------
# Save
# -----------------------------
logger.info("Saving adapter and tokenizer...")
model.save_pretrained("outputs-salt/adapter")
tok.save_pretrained("outputs-salt/tokenizer")
logger.info("✅ Saved adapter + tokenizer to outputs-salt/")
