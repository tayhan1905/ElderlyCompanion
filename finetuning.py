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
    logging as hf_logging,
)
from peft import LoraConfig, get_peft_model
import evaluate
from typing import Dict, List

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)7s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("singlish-lora")
hf_logging.set_verbosity_info()

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = "t5-small"  # (works well + fast); you can swap to "google/mt5-base" later
OUTPUT_DIR = "outputs-t5-lora"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128
SEED = 42

# -----------------------------
# Data
# -----------------------------
log.info("Loading dataset (Singlish→English)...")
ds = load_dataset("gabrielchua/singlish-to-english-synthetic")["train"].train_test_split(
    test_size=0.1, seed=SEED
)
train_ds, val_ds = ds["train"], ds["test"]
log.info(f"Train={len(train_ds)} | Val={len(val_ds)}")

def format_example(ex: Dict) -> Dict:
    # T5-style translation prompt
    # (Keep format stable between train & inference)
    src = f"translate Singlish to English: {ex['singlish']}"
    tgt = ex["english"]
    return {"src": src, "tgt": tgt}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(format_example,   remove_columns=val_ds.column_names)

# -----------------------------
# Tokenizer
# -----------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

def tok_fn(batch: Dict[str, List[str]]) -> Dict:
    model_inputs = tok(
        batch["src"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding=False,
    )
    with tok.as_target_tokenizer():
        labels = tok(
            batch["tgt"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

log.info("Tokenizing...")
train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["src","tgt"])
val_tok   = val_ds.map(tok_fn,   batched=True, remove_columns=["src","tgt"])

# -----------------------------
# Model + LoRA
# -----------------------------
log.info(f"Loading base model: {BASE_MODEL}")
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

# IMPORTANT: T5/mT5 LoRA targets by substring
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q","k","v","o","wi","wo"],  # correct for T5-family via substring match
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()  # should be > 0

# -----------------------------
# Training setup
# -----------------------------
collator = DataCollatorForSeq2Seq(tok, model=model)

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Some trainer versions pass tuples; handle both cases
    if isinstance(preds, tuple):
        preds = preds[0]
    preds_text = tok.batch_decode(preds, skip_special_tokens=True)
    labels = [[l] for l in tok.batch_decode(labels, skip_special_tokens=True)]
    bleu_res = bleu.compute(predictions=preds_text, references=labels)
    chrf_res = chrf.compute(predictions=preds_text, references=[r[0] for r in labels])
    return {"bleu": bleu_res["score"], "chrf": chrf_res["score"]}

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    logging_steps=20,
    report_to="none",
    fp16=torch.cuda.is_available(),   # safe on most consumer GPUs
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
# Train
# -----------------------------
log.info("🚀 Training...")
trainer.train()
log.info("✅ Training done.")

# -----------------------------
# Save adapter + tokenizer
# -----------------------------
adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
tok_dir = os.path.join(OUTPUT_DIR, "tokenizer")
model.save_pretrained(adapter_dir)
tok.save_pretrained(tok_dir)
log.info(f"Saved LoRA adapter → {adapter_dir}")
log.info(f"Saved tokenizer → {tok_dir}")

# -----------------------------
# Post-training sanity check
# -----------------------------
def translate(sentence: str, max_new_tokens=64) -> str:
    model.eval()
    prompt = f"translate Singlish to English: {sentence}"
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,   # use max_new_tokens to avoid chopping prompt
            num_beams=5,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True)

examples = [
    "wah the weather damn hot today leh",
    "you eat already or not?",
    "don’t play play ah, this one very expensive!",
    "later I go your house then we study together lah",
]

log.info("🔍 Sanity check:")
for s in examples:
    print(f"\n🗣️ Singlish: {s}")
    print(f"💬 English:  {translate(s)}")
