from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch 
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Load and prepare the dataset
dataset = load_dataset('csv', data_files={'train': 'annotated/annotated_train_data.csv', 'test': 'annotated/annotated_test_data.csv'})

# dataset = load_dataset("csv", data_files="annotated/annotated_train_data.csv")
dataset = dataset.rename_column("annotated_stance", "label")
dataset = dataset.rename_column("w", "text")
dataset = dataset.remove_columns(['Column1', 'Topic', 'Name', "Unnamed: 0"])
train_dataset = dataset["train"]
test_dataset = dataset["test"]

reddit_messages = test_dataset["text"]

print("Loaded data")

# Replace with your Hugging Face access token
access_token = ""

# Log in using the token
login(access_token)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = model.to(device)
print("Downloaded the model")

# Get the df ready 
prob_stance_df = pd.DataFrame(columns=["message", 'A', 'F', 'N'])

# Define system prompt 
system_prompt = """You are an expert in detecting stances in online discussions. 
   For each user message, determine the stance on the given topic.
    The topic is: abortion.
    Your response should be only one of the following tokens: [F, A, N].

    Examples:
    User: "I believe women should have the right to make decisions about their own bodies."
    Stance: F

    User: "Abortion is murder, and it should not be allowed under any circumstances."
    Stance: A

    User: "This issue is complicated, and I see both sides of the argument."
    Stance: N

    User: "Im in favor of all abortions, you misunderstand. I dont want any legal restrictions on abortions.
    Stance: F

    User: "What about the child? Does it have no rights? It is only there due to your selfish actions and should not be killed because you do not want it there."
    Stance: A 

    Now classify the following message:"""
for message in reddit_messages: 
   messages = [

            {

               "role": "system",

               "content": system_prompt

            },

            {"role": "user", "content": message},

         ]

   tokenizer.pad_token = tokenizer.eos_token

   tokenized_chat = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
   # print("tokenized_chat\n", tokenized_chat)

   inputs = tokenizer(tokenized_chat, return_tensors = "pt", padding = True).to(0)
   # print("inputs\n", inputs)


   output = model.generate(inputs["input_ids"], attention_mask = inputs["attention_mask"], pad_token_id = tokenizer.eos_token_id, max_new_tokens = 1, output_scores = True, return_dict_in_generate = True, renormalize_logits = True)
   # print("output\n", output)


   token_scores = torch.exp(output.scores[0][0]).cpu().numpy()
   # print("token_scores\n", token_scores)

   token_indexes = (token_scores > 0).nonzero()[0]
   # print("token_indexes\n", token_indexes)

   token_scores_output = dict(zip(map(tokenizer.decode, token_indexes), token_scores[token_scores > 0]))
   # print("token_scores\n", token_scores_output)

   # logits = output.scores[0][0]

   # Define the target tokens: 'A', 'F', 'N' and their respective token ids
   target_tokens = ['A', 'F', 'N']
   target_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in target_tokens]
   # print(target_token_ids)

   # Get the probabilities for tokens A, F, N
   token_probs_dict = {token: token_scores[token_id] for token, token_id in zip(target_tokens, target_token_ids)}

   # Print the probabilities for A, F, N
   # print("Probabilities for tokens A, F, N:", token_probs_dict)

   token_probs_dict["message"] = message 
   row_df = pd.DataFrame([token_probs_dict])
   prob_stance_df = pd.concat([prob_stance_df, row_df], ignore_index=True)

   # Clear cache 
   torch.cuda.empty_cache()

print(prob_stance_df)
prob_stance_df.to_csv("output/llama_probabilities.csv", index=False)

# getting the accuracy 
prob_stance_df['stance'] = prob_stance_df[['A','F', "N"]].idxmax(axis=1)
dict_labels = {"A": "against", "F": "for", "N": "neutral"}
prob_stance_df["stance"] = prob_stance_df["stance"].map(dict_labels)
matching_rows = (prob_stance_df["stance"]==test_dataset["label"]).sum()
accuracy = matching_rows/test_dataset.num_rows
print("saving accuracy")
df_metrics = pd.DataFrame([accuracy])
df_metrics.to_csv("output/llama_accuracy.csv", index=False)

#confusion matrix
conf_matrix = confusion_matrix(test_dataset["label"], prob_stance_df["stance"], labels=['for', 'against', 'neutral'])
print(conf_matrix)

#logging prompt, accuracy and confusion matrix
with open("classification_log.txt", "a") as file:
    file.write(f"Prompt: {system_prompt}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Confusion Matrix:\n{conf_matrix}\n")
    file.write("-" * 40 + "\n")  # Separator for readability


