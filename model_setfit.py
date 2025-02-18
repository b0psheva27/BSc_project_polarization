# Import necessary libraries
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import pandas as pd

# Step 1: Load and prepare the dataset
dataset = load_dataset("csv", data_files="annotated/annotated_data.csv")
dataset = dataset.rename_column("annotated_stance", "label")
dataset = dataset.rename_column("w", "text")
dataset = dataset.remove_columns(['Column1', 'Topic', 'Name'])

train_dataset = dataset["train"]

# Map labels
label_map = {"for": 0, "neutral": 1, "against": 2}
train_dataset = train_dataset.map(lambda row: {"label": label_map[row["label"]]})

# Step 2: Load the SetFit model
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")

# Step 3: Define training arguments
args = TrainingArguments(
    batch_size=4,
    num_epochs=1,
)

# Step 4: Initialize the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

# Step 5: Train the model
trainer.train()

# Step 6: evaluate performance

trainer.evaluate(test_dataset)

# Step 6: Make predictions on the training dataset
# predictions = trainer.predict(train_dataset)

# # Step 7: Save predictions
# predicted_labels = predictions.predictions.argmax(axis=1)

# # Convert logits to DataFrame
# logits_df = pd.DataFrame(predictions)

# # Save to CSV
# logits_df.to_csv("logits.csv", index=False)

# # Convert to a DataFrame for easier inspection and saving
# predictions_df = pd.DataFrame({
#     "text": train_dataset["text"],
#     "true_label": train_dataset["label"],
#     "predicted_label": predicted_labels
# })

# # Save the predictions to a CSV file
# predictions_df.to_csv("predictions.csv", index=False)

# # Step 8: Calculate model performance (classification report)
# true_labels = train_dataset["label"]
# model_performance = classification_report(true_labels, predicted_labels, target_names=["for", "neutral", "against"])

# # Save the performance metrics to a text file
# with open("model_performance.txt", "w") as f:
#     f.write(model_performance)

# print("Training completed and results saved.")
