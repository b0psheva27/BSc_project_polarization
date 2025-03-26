# BSc_project_polarization

## Folders
models - the saved bertopic model with all weigths and configs
archive - old scripts (not part of main pipeline)
month_data - the dataset for the whole month of December messages. Contains also messages from users who write too few messages to be considered statistically significant, so they may not be part of the final user network.
notebooks - main code notebooks for different stages of project

## Files
combined_network.csv - edgelist for the network containing users who tallk about our 3 topics. From it the LCC needs to be taken. Total number of nodes in LCC is 511. This is the number of users we will be working with.
network_users_combined - list of users and node ids for our network of users for the 3 topics
