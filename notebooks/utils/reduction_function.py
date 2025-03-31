# imports 
import random 
from sklearn.decomposition import PCA
import sklearn.preprocessing as pre
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import umap


# # get the stance 
# def get_stance(row): #old version
#     if row["N"] == 1.0: 
#         return 0
#     elif row["A"] > row["F"]:
#         return row["A"]
#     elif row["F"] > row["A"]:
#         return row["F"]*-1
#     else: 
#         return random.choice([row["A"], row["F"]])
    
def get_stance(row): #new version that handles case of all 0 probs
    if row["N"] == 1.0:
        return 0
    elif row["N"] == row["A"] == row["F"] == 0:
        return 0
    elif row["A"] > row["F"]:
        return row["A"]
    elif row["F"] > row["A"]:
        return row["F"]*-1
    else:
        return random.choice([row["A"], row["F"]])

def get_avg(data, topic):
    data_avg = data.groupby("user")[['A', 'F', 'N']].mean().reset_index()
    data_avg[f"stance_{topic}"] = data_avg.apply(lambda x: get_stance(x), axis = 1)
    return data_avg

# merge the datasets 
def merge_datasets(data1, data2, data3): 
    merged_df = pd.merge(pd.merge(data1, data2, on='user'), data3, on='user')
    return merged_df 

# PCA 
def apply_pca(merged_df, colume_list): 
    # scaler
    scaler = pre.StandardScaler()
    merged_df[colume_list] = scaler.fit_transform(merged_df[colume_list])

    # pca 
    pca = PCA(n_components=1)
    merged_df["pca_component"] = pca.fit_transform(merged_df[colume_list])
    return merged_df

# UMAP 
def apply_umap(merged_df, column_list):
    # scaler 
    scaler = pre.StandardScaler()
    merged_df[column_list] = scaler.fit_transform(merged_df[column_list])

    # umap 
    reducer_1D = umap.UMAP(n_components=1, random_state=42)
    merged_df["umap_component"] = reducer_1D.fit_transform(merged_df[column_list])
    return merged_df

# Plots 
def plot_c(merged_df, pc_column, color_topic ): 
    num_users = merged_df.shape[0]
    y_axis = np.random.rand(num_users)
    scatter = plt.scatter(merged_df[pc_column], y_axis, c=merged_df[color_topic], cmap='coolwarm')
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{color_topic} stance')
    if pc_column == "umap_component": 
        plt.xlabel('UMAP component')
    elif pc_column == "pca_component": 
        plt.xlabel('Principal component')
    plt.ylabel('Random')
    plt.title('Scatter plot of the principal component')
    plt.show()

 
def plot_c_subplot(merged_df, pc_column, color_topics):
    """Function to make a subplot of scatterplots for all reduced component vs random numbers,
      colored by stances on each of the 3 topics
      
      color_topics: list of column names (strings) where the stances are -> ["stance_abortion", "stance_marriage", "stance_political"]
    """
    num_plots = len(color_topics)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), sharey=True)
 
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one plot
 
    for i, (color_topic, ax) in enumerate(zip(color_topics, axes)):
        num_users = merged_df.shape[0]
        y_axis = np.random.rand(num_users)
 
        scatter = ax.scatter(merged_df[pc_column], y_axis, c=merged_df[color_topic], cmap='coolwarm')
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(f'{color_topic} stance')
 
        ax.set_xlabel('UMAP component' if pc_column == "umap_component" else 'Principal component')
        if i == 0:
            ax.set_ylabel('Random')  # Label only on first plot
        ax.set_title(f'Scatter plot - {color_topic}')
 
    plt.tight_layout()
    plt.show()
