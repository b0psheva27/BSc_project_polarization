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

topic_model_fitted = BERTopic.load("models/bert_topic_model")

print("Model loaded... start fitting")

# fitting the bertopic model 
# topic_model_fitted = topic_model.fit(messages_list)

topic_model_fitted = topic_model_fitted.reduce_topics(messages_list, nr_topics=1000)

# get info on topics (names, important words, representative reddit message/document) !!!before reclustering
topic_info = topic_model_fitted.get_topic_info() #topics with their indeces and repr words
topic_info.to_csv("output/reclustered_weekly_topics.csv", sep="\t", index=False)

weekly_topics = topic_model_fitted.topics_  #topics per message
# Store topics in the original DataFrame
messages_df['topic'] = weekly_topics

# Create a mapping of topic ID to topic name
topic_id_to_name = topic_info.set_index("Topic")["Name"].to_dict()

# Map topic IDs to topic names in the DataFrame
messages_df["topic_name"] = messages_df["topic"].map(topic_id_to_name)
messages_df.to_csv("output/reclustered_weekly_data.csv", sep=",")

print("Saving model...")
topic_model_fitted.save("models/reclustered_bert_topic_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
print("Model saved!")

print("############## Transforming new data ################")
monthly_messages_df = pd.read_csv("month_data/22_12.csv", sep="\t")

# Transform the text column (no need to refit)
topics, probs = topic_model_fitted.transform(monthly_messages_df['text'].astype(str).tolist())

# Store topics in the original DataFrame
monthly_messages_df['topic'] = topics

# Create a mapping of topic ID to topic name
topic_id_to_name = topic_info.set_index("Topic")["Name"].to_dict()

# Map topic IDs to topic names in the DataFrame
monthly_messages_df["topic_name"] = monthly_messages_df["topic"].map(topic_id_to_name)

print("Saving the topics per message for December 2022...")
monthly_messages_df.to_csv("output/reclustered_monthly_data.csv", sep=",")
# We don't save the topics with their ids and repr words/docs for the monthly data, since they are supposed to be the same as the weekly ones 
# (the model is fitted on the weekly data only)

print("Done!")

