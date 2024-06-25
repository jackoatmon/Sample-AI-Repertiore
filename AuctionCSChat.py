import os
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Necessary packages for integration: transformers, torch, pandas, datasets

# loading env variables
data_path = os.getenv("data_path", "./NLP Data/AuctionCS-Questions-Answers.csv")
output_dir = os.getenv("output_dir", "./AuctionCS Outputs")
model_path = os.getenv("model_path", "./AuctionCS Model")

# loading model & tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# loading dataset
data = pd.read_csv(data_path)
dataset = Dataset.from_pandas(data)

# preprocessing dataset
def preprocess_function(examples):
    inputs = [ex for ex in examples['Question']]
    targets = [ex for ex in examples['Answer']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# setting training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# initializing & calling trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

# saving new model & tokenizer
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
