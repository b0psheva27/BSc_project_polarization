# getting the data 
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': 'annotated/annotated_train_data.csv', 'test': 'annotated/annotated_test_data.csv'})

# dataset = load_dataset("csv", data_files="annotated/annotated_train_data.csv")
dataset = dataset.rename_column("annotated_stance", "label")
dataset = dataset.rename_column("w", "text")
dataset = dataset.remove_columns(['Column1', 'Topic', 'Name', "Unnamed: 0"])
train_dataset = dataset["train"]
test_dataset = dataset["test"]

label_map = {"for": 0, "neutral": 1, "against": 2}
train_dataset = train_dataset.map(lambda row: {"label": label_map[row["label"]]})
test_dataset = test_dataset.map(lambda row: {"label": label_map[row["label"]]})

