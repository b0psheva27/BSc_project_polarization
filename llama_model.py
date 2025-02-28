from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch 
from datasets import load_dataset
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

reddit_messages = test_dataset["text"]

print("Loaded data")

# Replace with your Hugging Face access token
access_token = "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"

# Log in using the token
login(access_token)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloaded the model")

# Get the df ready 
prob_stance_df = pd.DataFrame(columns=["message", 'A', 'F', 'N'])

# Define system prompt 
system_prompt = """You are an expert in detecting stances in online discussions. 
   For each user message, determine the stance on the given topic.
    The topic is: abortion.
    Your response should be only one of the following tokens: [for, against, neutral].

    Examples:
    User: "I believe women should have the right to make decisions about their own bodies."
    Stance: for

    User: "Abortion is murder, and it should not be allowed under any circumstances."
    Stance: against

    User: "This issue is complicated, and I see both sides of the argument."
    Stance: neutral

    Now classify the following message:"""

for message in reddit_messages: 
   messages = [

            {

               "role": "system",

               "content": system_prompt

            },

            {"role": "user", "content": message},

         ]

   tokenized_chat = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
   print("tokenized_chat\n", tokenized_chat)

   inputs = tokenizer(tokenized_chat, return_tensors = "pt", padding = True).to(0)
   print("inputs\n", inputs)


   output = model.generate(inputs["input_ids"], attention_mask = inputs["attention_mask"], pad_token_id = tokenizer.eos_token_id, max_new_tokens = 1, output_scores = True, return_dict_in_generate = True, renormalize_logits = True)
   print("output\n", output)


   token_scores = torch.exp(output.scores[0][0]).cpu().numpy()
   print("token_scores\n", token_scores)


   token_indexes = (token_scores > 0).nonzero()[0]
   print("token_indexes\n", token_indexes)

   token_scores = dict(zip(map(tokenizer.decode, token_indexes), token_scores[token_scores > 0]))
   print("token_scores\n", token_scores)

