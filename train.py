import torch
import os
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# 1. MODEL VE İŞLEMCİ KURULUMU
MODEL_ID = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="turkish", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# [DÜZELTME] Generation Config Temizliği
if hasattr(model.config, "suppress_tokens"):
    del model.config.suppress_tokens
if hasattr(model.config, "forced_decoder_ids"):
    del model.config.forced_decoder_ids

model.generation_config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.gradient_checkpointing_disable() 

# 2. VERİ SETİ HAZIRLIĞI VE BÖLME
dataset = load_dataset("audiofolder", data_dir="./processed_dataset", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Veriyi %90 Train, %10 Eval olarak bölüyoruz
dataset = dataset.train_test_split(test_size=0.1)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

encoded_train = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=8)
encoded_eval = dataset["test"].map(prepare_dataset, remove_columns=dataset["test"].column_names, num_proc=8)

# 3. METRİK HESAPLAMA (WER)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 4. DATA COLLATOR
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. EĞİTİM PARAMETRELERİ
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-final-model",
    per_device_train_batch_size=64, 
    gradient_accumulation_steps=1, 
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=5000, 
    bf16=True, 
    tf32=True, 
    optim="adamw_torch_fused",
    
    # Eval ayarları (Aktif etmek istersen 'no'yu 'steps' yap)
    eval_strategy="no", 
    save_steps=500,
    logging_steps=10,
    report_to=["tensorboard"],
    dataloader_num_workers=8,
    load_best_model_at_end=False,
    
    # Hub ayarları
    push_to_hub=True,
    hub_model_id="meet-ali-123/deneme-ses",# hugginface yolu alcaz
    hub_strategy="checkpoint",
)

# 6. TRAINER BAŞLATMA
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=encoded_train,
    eval_dataset=encoded_eval,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./whisper-final-model")
processor.save_pretrained("./whisper-final-model")