# Imports
from bertopic import BERTopic
import pandas as pd
import numpy as np
import nltk 
import matplotlib.pyplot as plt

# getting weekly data 
messages_df = pd.read_csv("reddit_22_51/messages.csv", sep="\t")
# convert to list of strings (input needed by bertopic model)
messages_list = messages_df["text"].astype(str).tolist()
topic_model = BERTopic()

print("Model loaded... start fitting")

# fitting the bertopic model 
topic_model_fitted = topic_model.fit(messages_list)

print("Model fitted... get topic info")
# get info on topics (names, important words, representative reddit message/document)
topic_info = topic_model_fitted.get_topic_info()
topic_info.to_csv("output/weekly_topics.csv", sep="\t", index=False)

print("Saving model...")
topic_model_fitted.save("models/bert_topic_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
print("Model saved!")

print("############## Transforming new data ################")
monthly_messages_df = pd.read_csv("month_data/22_12.csv", sep="\t")

# Transform the text column (no need to refit)
topics, probs = topic_model_fitted.transform(monthly_messages_df['text'].tolist())

# Store topics in the original DataFrame
monthly_messages_df['topic'] = topics

# Create a mapping of topic ID to topic name
topic_id_to_name = topic_info.set_index("Topic")["Name"].to_dict()

# Map topic IDs to topic names in the DataFrame
monthly_messages_df["topic_name"] = monthly_messages_df["topic"].map(topic_id_to_name)

print("Saving the monthly topics...")
monthly_messages_df.to_csv("output/montly_topics.csv", sep=",")

print("Done!")

