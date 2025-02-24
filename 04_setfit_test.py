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
    batch_size=32,
    num_epochs=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

trainer.train()

metrics = trainer.evaluate(test_dataset)

print("saving metrics")
df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv("output/metrics.csv", index=False)

print("saving model")
# Step 6: Save the model 
model.save_pretrained("models/setfit_model")

print("done")