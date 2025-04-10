import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

# ------------------------------------------------------------------------
# 2. Load Datasets and Create DatasetDict
# ------------------------------------------------------------------------

train_df = pd.read_csv("ft_train_masked.csv")
train_df.drop(columns=["Unnamed: 0"], inplace=True)

val_df = pd.read_csv("ft_valid_masked.csv")
val_df.drop(columns=["Unnamed: 0"], inplace=True)

test_df = pd.read_csv("ft_test_masked.csv")
test_df.drop(columns=["Unnamed: 0"], inplace=True)

# create DatasetDict from data (so that it can be passed to the trainer later)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})


# ------------------------------------------------------------------------
# 3. Load Pre-trained Model & Tokenizer
# ------------------------------------------------------------------------

print("STARTING STEP 3")
model_checkpoint = "Salesforce/codet5-small"

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# load pre-trained tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

# add tokens <MASK> and <TAB> 
tokenizer.add_tokens(["<MASK>"]) 
tokenizer.add_tokens(["<TAB>"]) # comment out for inputs with NO <TAB> token

model.resize_token_embeddings(len(tokenizer))


# ------------------------------------------------------------------------------------------------
# 4. We prepare now the fine-tuning dataset using the tokenizer we preloaded
# ------------------------------------------------------------------------------------------------

print("STARTING STEP 4")
def preprocess_function(examples):
    inputs = examples["masked_with_tab"]
    # inputs = examples["masked_no_tab"] 
    targets = examples["target_block"]

    # tokenize the inputs
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # tokenize the target
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)


# ------------------------------------------------------------------------
# 5. Define Training Arguments and Trainer
# ------------------------------------------------------------------------

print("STARTING STEP 5")
training_args = TrainingArguments(
    output_dir="./codet5-finetuned",
    eval_strategy="epoch", 
    save_strategy="epoch", # creates checkpoint every epoch
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
    report_to="wandb" # use wandb to track training process
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # stop training when loss increases twice on the validation set
)


# ------------------------
# 6. Train the Model
# ------------------------

print("STARTING TRAINING")
trainer.train()

# save the model to a folder called final_model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")