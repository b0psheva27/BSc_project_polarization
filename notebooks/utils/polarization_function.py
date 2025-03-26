import networkx as nx
import numpy as np
from scipy.special import binom
from scipy.sparse import csgraph
import pandas as pd 

# Function to calculate the pseudoinverse of the Laplacian of the network
def _ge_Q(network):
    A = nx.adjacency_matrix(network).todense().astype(float)
    return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed=False))

def ge(src, trg, network, Q=None):
    """Calculate GE for network.

    Parameters:
    ----------
    srg: vector specifying node polarities
    trg: vector specifying node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError("""Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one.""")
    src = np.array([src[n] if n in src else 0. for n in network.nodes()])
    trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
    diff = src - trg
    if Q is None:
        Q = _ge_Q(network)

    ge_dist = diff.T.dot(np.array(Q).dot(diff))

    if ge_dist < 0:
        ge_dist = 0

    return np.sqrt(ge_dist)

# This function reads the edgelist and the opinion value
# It returns the delta polarization score
def calc_pol(path):
   G = nx.read_edgelist(f"{path}/edgelist.csv", delimiter = ',')
   o = {}
   with open(f"{path}/user_scores.csv", 'r') as f:
      for line in f:
         fields = line.strip().split(',')
         if fields[0] in G.nodes:
            o[fields[0]] = float(fields[1])
   return ge(o, {}, G)


def calc_pol2(path_edgelist, path_user_scores ):
   df = pd.read_csv(f"{path_edgelist}", sep="\t")  # Read CSV into DataFrame
   G = nx.from_pandas_edgelist(df, "src", "trg")
   print(df.head())
   print(G)
   print(G.nodes)
   o = {}
#    df_user_scores = pd.read_csv(f"{path_user_scores}", sep="\t")
   with open(f"{path_user_scores}", 'r') as f:
      next(f)
      for line in f:
         fields = line.strip().split(',')
         print(fields)
         if int(fields[0]) in G.nodes:
            o[int(fields[0])] = float(fields[1])
   print(o)
   return ge(o, {}, G)




def calc_pol3(path_edgelist, path_user_scores, component):
   df = pd.read_csv(f"{path_edgelist}", sep="\t")  # Read CSV into DataFrame
   G = nx.from_pandas_edgelist(df, "src", "trg")
   print(df.head())
   print(G)
   print(G.nodes)
   o = {}
   df_user_scores = pd.read_csv(f"{path_user_scores}", sep=",")
   o = df_user_scores.set_index("nodeid")[f"{component}"].to_dict()
   print(o)
   return ge(o, {}, G)