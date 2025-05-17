import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch 
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pandas as pd
import csv
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Load and prepare the dataset
monthly_topic_data = pd.read_csv("month_data/reclustered_monthly_data_final.csv", sep="\t", encoding = "utf-8", quoting = csv.QUOTE_NONE)
political_messages = monthly_topic_data[monthly_topic_data["topic"]==1]
# user_names = political_messages["user"].to_list()
# post_id = political_messages["post_id"].to_list()
# id_ = political_messages["id"].to_list()
# reddit_messages = political_messages["text"].to_list()
political_messages["A"] = np.nan 
political_messages["F"] = np.nan 
political_messages["N"] = np.nan 

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
# prob_stance_df = pd.DataFrame(columns=["message", 'A', 'F', 'N'])

# Define system prompt
system_prompt = """You are an expert in detecting stances in online discussions.
For each user message, classify their political stance into one of three categories:
  - F (Liberal): If the user expresses liberal or left leaning views, speaks positively about liberal views, or speaks negatively of conservative views
  - A (Conservative): If the user expresses conservative or right leaning views, speaks positively about conservative views, or speaks negatively of liberal views
  - N (Neutral): Only if the user expresses no opinion on politics i.e. no liberal and no conservative views 

Examples:
Message: "You come across as a miserable conservative who hates everyone, and I am not that."
Output: F

Message: "Liberalism is lame. It’s for idiots"
Output: A

Message: "Modern politics in our country operate on a foundation of a binary left-right spectrum"
Output: N

Your output must be strictly one of the following tokens: [F, A, N].
Now classify the following user message:
    """
for idx, row in political_messages.iterrows(): 
   message = row["text"]
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

   prob_A = token_probs_dict["A"]
   prob_F = token_probs_dict["F"]
   prob_N = token_probs_dict["N"]

   political_messages.at[idx, "A"] = prob_A
   political_messages.at[idx, "F"] = prob_F
   political_messages.at[idx, "N"] = prob_N

   # Print the probabilities for A, F, N
   # print("Probabilities for tokens A, F, N:", token_probs_dict)

#    token_probs_dict["message"] = message 
#    row_df = pd.DataFrame([token_probs_dict])
#    prob_stance_df = pd.concat([prob_stance_df, row_df], ignore_index=True)

   # Clear cache 
   torch.cuda.empty_cache()

print(political_messages)
political_messages.to_csv("output/llama_probabilities_full_political.csv", index=False, sep = "\t")

print("done")
