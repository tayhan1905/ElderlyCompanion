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
# Model + LoRA (SALT style)
# -----------------------------
logger.info(f"Loading base model {base_model}...")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=torch.float32)

# Correct LoRA configuration for mT5
logger.info("Configuring LoRA adapters for mT5...")
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",
        "DenseReluDense.wi",
        "DenseReluDense.wo"
    ],
    bias="none",
    task_type="SEQ_2_SEQ_LM"
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
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    bf16=torch.cuda.is_available(),
    report_to="none",
    logging_dir="logs"
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
# Verify LoRA attachment
# -----------------------------
model.print_trainable_parameters()

# -----------------------------
# Save adapter + tokenizer
# -----------------------------
adapter_dir = "outputs-salt/adapter"
tok_dir = "outputs-salt/tokenizer"
logger.info("Saving adapter and tokenizer...")
model.save_pretrained(adapter_dir)
tok.save_pretrained(tok_dir)
logger.info(f"✅ Saved adapter to {adapter_dir} and tokenizer to {tok_dir}")

# -----------------------------
# Inference sanity check
# -----------------------------
logger.info("🔍 Running post-training inference sanity check...")

def translate_singlish(sentence, max_length=128):
    prompt = f"Translate Singlish to English:\nSinglish: {sentence}\nEnglish:"
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
    translation = tok.decode(outputs[0], skip_special_tokens=True)
    return translation

# Example sentences to verify translation quality
examples = [
    "wah the weather damn hot today leh",
    "you eat already or not?",
    "don’t play play ah, this one very expensive!",
    "later I go your house then we study together lah",
]

for ex in examples:
    print(f"\n🗣️ Singlish: {ex}")
    print(f"💬 English: {translate_singlish(ex)}")

logger.info("✅ Inference sanity check complete.")
