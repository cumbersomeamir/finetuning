#!pip3 install transformers datasets numpy pandas evaluate
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification , TrainingArguments , Trainer
import numpy as np
import evaluate


#Loading the Dataset
dataset = load_dataset("yelp_review_full")
print("The Dataset is", dataset["train"][100])
print("The Dataset type is", type(dataset))


#Loading the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Defining the Tokenize function
def tokenize_function(examples):
  return tokenizer(examples["text"], padding= max_length, truncation=True)

#Mapping the Tokenized dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)


#Training a small dataset to save time
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#Loading the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels =5)

#Defining the accuracy metric
metric = evaluate.load("accuracy")

#Defining the compute metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#Initialising the traning arguments
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

#Creating the trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

#Training the model
trainer.train()
