from datasets import load_dataset
from setfit import sample_dataset
from setfit import SetFitModel
from setfit import TrainingArguments
from setfit import Trainer
import pandas as pd 


model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")

dataset = load_dataset("SetFit/sst2")
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
test_dataset = dataset["test"]

model.labels = ["negative", "positive"]

args = TrainingArguments(
    batch_size=4,
    num_epochs=(1,16),
)

# Step 4: Initialize the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("starting training")
# Step 5: Train the model
trainer.train()
print("training finished")

# Step 7: evaluate performance

metrics = trainer.evaluate(test_dataset)

print("saving metrics")
df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv("output/metrics.csv", index=False)

print("saving model")
# Step 6: Save the model 
model.save_pretrained("models/setfit_model")

print("done")
