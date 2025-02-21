# Import necessary libraries
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import pandas as pd

# Step 1: Load and prepare the dataset
dataset = load_dataset('csv', data_files={'train': 'annotated/annotated_train_data.csv', 'test': 'annotated/annotated_test_data.csv'})

# dataset = load_dataset("csv", data_files="annotated/annotated_train_data.csv")
dataset = dataset.rename_column("annotated_stance", "label")
dataset = dataset.rename_column("w", "text")
dataset = dataset.remove_columns(['Column1', 'Topic', 'Name', "Unnamed: 0"])
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# map labels 
label_map = {"for": 0, "neutral": 1, "against": 2}
train_dataset = train_dataset.map(lambda row: {"label": label_map[row["label"]]})
test_dataset = test_dataset.map(lambda row: {"label": label_map[row["label"]]})

# experiment 
# train_dataset_5 = train_dataset.select(range(5)) 
# test_dataset_5 = test_dataset.select(range(5)) 


# Step 2: Load the SetFit model
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")

# Step 3: Define training arguments
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

# Step 8: Make predictions on the training dataset
texts = test_dataset["text"]
predictions = model.predict(texts)

# add into dataframe
df_predictions = pd.DataFrame(predictions, columns=["Class"])

# Add original texts
df_predictions.insert(0, "Text", texts)

# Save to CSV (optional)
df_predictions.to_csv("output/predictions.csv", index=False)

# Step 9: Get probabilities 
probabilities = model.predict_proba(texts)

df_probs = pd.DataFrame(probabilities, columns=[f"Prob_Class_{i}" for i in range(probabilities.shape[1])])

# Add original texts
df_probs.insert(0, "Text", texts)

# Save to CSV (optional)
df_probs.to_csv("output/probabilities.csv", index=False)

print("saving model")
# Step 6: Save the model 
model.save_pretrained("models/setfit_model")

print("done")
