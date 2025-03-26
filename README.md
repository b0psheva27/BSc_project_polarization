# BSc_project_polarization

## Folders
models - the saved bertopic model with all weigths and configs

archive - old scripts (not part of main pipeline)

month_data - the dataset for the whole month of December messages. Contains also messages from users who write too few messages to be considered statistically significant, so they may not be part of the final user network.

notebooks - main code notebooks for different stages of project

output_hpc - files from the models and scripts on the HPC

output_network - files connected to the network of users

## Files

combined_network.csv - edgelist for the network containing users who tallk about our 3 topics. From it the LCC needs to be taken. Total number of nodes in LCC is 511. This is the number of users we will be working with.

network_users_combined - list of users and node ids for our network of users for the 3 topics

network_filtered.csv - the edgelist for the LCC of the combined network. This is the file that needs to be loaded when working with the relevant users.

user_scores_pca/umap.csv - dummy datasets for the implementation of the polarization code
