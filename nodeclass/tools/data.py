#!/usr/bin/python3
# coding: utf-8

# Hierarchical node classification (attribute prediction considering hierarchy of classes)
# Data preprocessing
# Miguel Romero, 2021 ago 12

import re
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import multiprocessing
from tqdm import tqdm
from collections import deque

from .mst import *

# Node embedding
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation

# Cross-validation and scaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


def nodes_in_bfs(bfs, root):
  """
  Convert a bfs object to a list

  :param bfs: bfs object from networkx
  :type bfs: object
  :param root: id of the root in the bfs
  :type root: int

  :return: list of nodes in the subgraph in bfs order
  :rtype: list[int]
  """
  nodes = sorted(list(set([u for u,v in bfs] + [v for u,v in bfs])))
  nodes = np.setdiff1d(nodes, [root]).tolist()
  nodes = [root] + nodes
  return nodes


def create_matrices(edgl, isa, cls_def, n2c, output_path, trace=False):
  """
  Create the adjacency and association matrices of the nodes and classes from
  the edglists for the graph and the associations between classes.

  :param edgl: Edgelist of the graph with columns 'Source' and 'Target'
  :type edgl: Dataframe
  :param isa: Edgelist of the initial representations of the
    associations between classes with columns 'Class' and 'Ancestor'
  :type isa: DataFrame
  :param cls_def: Description of the classes with columns 'Class' and 'Desc'
  :type cls_def: DataFrame
  :param n2c: Associations between classes and nodes by pairs '(node, class)'
    with columns 'Node' and 'Class'
  :type n2c: DataFrame
  :param output_path: Path where the output is stored
  :type output_path: string
  :param trace: Flag to print the trace of the process
  :type trace: boolean

  :return: Adjacency matrix of the nodes, adjacency matrix of the classes,
    association matrix between nodes and classes, list of nodes and list of
    classes in the same order of the matrices
  :rtype: np.matrix[float] *MxM*, np.matrix[float] *NxN*,
    np.matrix[float] *MxN*, np.array[string] *M*, np.array[string] *N*
  """

  V = np.array(sorted(list(set(edgl['Source'].tolist()+edgl['Target'].tolist()))))
  nV, idxV = len(V), dict([(v,i) for i,v in enumerate(V)])

  C = np.array(sorted(list(set(cls_def['Class'].tolist()+n2c['Class'].tolist()))))
  nC, idxC = len(C), dict([(c,i) for i,c in enumerate(C)])

  ####################
  # 1. Create matrices
  ####################

  # Adj matrix
  # nV:number of nodes, idxV:node index map
  adj = np.zeros((nV,nV))
  for edge in tqdm([tuple(x) for x in edgl.to_numpy()]):
    u, v = idxV[edge[0]], idxV[edge[1]]
    adj[u][v] = adj[v][u] = 1

  # Adj matrix of classes
  # nC:number of classes, idxC:class index map
  cl_adj = np.zeros((nC,nC))
  for edge in tqdm([tuple(x) for x in isa.to_numpy()]):
    u, v = idxC[edge[0]], idxC[edge[1]]
    cl_adj[u,v] = 1

  # compute the transitive closure of the ancestor of a class (idx)
  def ancestors(cls):
    tmp = np.nonzero(cl_adj[cls,:])[0].tolist()
    ancs = list()
    while len(tmp) > 0:
      tmp1 = list()
      for i in tmp:
        ancs.append(i)
        tmp1 += np.nonzero(cl_adj[i,:])[0].tolist()
      tmp = list(set(tmp1))
    return ancs

  # node by class matrix
  node_by_cl = np.zeros((nV,nC))
  for edge in tqdm([tuple(x) for x in n2c.to_numpy()]):
    u, v = idxV[edge[0]], idxC[edge[1]]
    node_by_cl[u,v] = 1
    node_by_cl[u,ancestors(v)] = 1

  if trace:
    print('**Matrices**')
    print('Nodes: \t{0:6}'.format(len(adj)))
    print('Nodes labeled: \t{0:6}'.format(np.count_nonzero(node_by_cl)))
    print('Node interactions: \t{0:6.0f}'.format(np.sum(adj)/2))
    print('Classes: \t{0:6}'.format(len(cl_adj)))
    print('Classes relationships: \t{0:6.0f}'.format(np.sum(cl_adj)))

  return adj, cl_adj, node_by_cl, V, C


def generate_hierarchy(adj, cl_adj, node_by_cl, cls_def, V, C, output_path, filter=None, trace=False):
  """
  Find the possible sub-hierarchies of the hieararchy of classes by removing
  the isolated classes and applying a filter (if required) to the number of
  nodes associated from the adjacency and association matrices of graph and
  classes.

  :param adj: Adjacency matrix of the nodes
  :type adj: np.matrix[float] *MxM*
  :param cl_adj: Adjacency matrix of the classes
  :type cl_adj: np.matrix[float] *NxN*
  :param node_by_cl: Association matrix between nodes and classes
  :type node_by_cl: np.matrix[float] *MxN*
  :param cls_def: Description of the classes with columns 'Class' and 'Desc'
  :type cls_def: DataFrame
  :param V: List of nodes in the same order of the 'adj' matrix
  :type V: np.array[string] *M*
  :param C: List of classes in the same order of the 'cl_adj' matrix
  :type C: np.array[string] *N*
  :param output_path: Path where the output is stored
  :type output_path: string
  :param filter: Filter applied to the classes to be used for prediction from
    the hierarchy, lower and upper bound of the number of nodes associated to
    the classes, default to None
  :type filter: Tuple[int, int]
  :param trace: Flag to print the trace of the process, default to False
  :type trace: boolean

  :return: List of roots (class index) of each sub-hierarchy from the hierarchy
    of classes, list of classes (indexes) within each sub-hiearchy (the first
    element of the list is the root class)
  :rtype: np.array[int], np.array[np.array[int]]
  """

  nC, idxC = len(C), dict([(c,i) for i,c in enumerate(C)])

  pred_clss_idx = np.arange(nC)
  if filter is not None:
    # Prune classes according if required, very specific and extremes with little
    # to no information classes are avoided. Select classes used for prediction
    # Accoding to restriction 5 <= nodes annotated <= 300
    filt_clss_idx = list()
    for i in range(nC):
      if filter[0] <= np.count_nonzero(node_by_cl[:,i]) <= filter[1]:
        filt_clss_idx.append(i)

    # Including the ancestor of the selected classes
    pred_clss_idx = list(filt_clss_idx)
    for tidx in filt_clss_idx:
      pred_clss_idx += np.nonzero(cl_adj[tidx,:])[0].tolist()
    pred_clss_idx = np.array(sorted(list(set(pred_clss_idx))))

  # Subgraph from classes to predict for hiearchy creation
  sub_cl_adj = cl_adj[np.ix_(pred_clss_idx,pred_clss_idx)].copy()
  sub_cl_adj_edgelist = np.transpose(np.nonzero(np.transpose(sub_cl_adj))).tolist()
  subg_nx = nx.DiGraph()
  subg_nx.add_nodes_from(np.arange(len(pred_clss_idx)))
  subg_nx.add_edges_from(sub_cl_adj_edgelist)

  # find possible root classes in go subgraph
  roots_idx = list()
  for tidx, cls in enumerate(pred_clss_idx):
    if np.count_nonzero(sub_cl_adj[tidx,:]) == 0: # classes wo ancestors
      roots_idx.append(tidx)
  roots_idx = np.array(roots_idx) # list of roots in the hiearchy

  # detect isolated clss and create sub-hierarchies
  subh_pred_clss = list() # number of classes to predict in each sub-hiearchy
  _roots_idx = list()
  for root in roots_idx:
    bfs = nx.bfs_tree(subg_nx, root).edges()

    if len(bfs) > 0: # if no isolated cls
      _roots_idx.append(pred_clss_idx[root])
      bfs_nodes = pred_clss_idx[nodes_in_bfs(bfs, root)]
      subh_pred_clss.append(bfs_nodes)

  roots_idx = _roots_idx

  # list sub-hierarchies
  df_subh = pd.DataFrame(columns=['Root_idx', 'Root','Classes','Nodes','Desc'])
  for i, tidx in enumerate(roots_idx):
    cls = C[tidx]
    data = [tidx, cls]
    data += [len(subh_pred_clss[i])] # number of clss to predict in sub-hier.
    data += [np.count_nonzero(node_by_cl[:,idxC[cls]])] # number of nodes in sub.
    data += [cls_def[cls_def['Class']==cls]['Desc'].tolist()[0]]
    df_subh.loc[i] = data

  df_subh = df_subh.sort_values(by=['Classes','Nodes'], ascending=False).reset_index(drop=True)
  df_subh.to_csv('{0}/hierarchies.csv'.format(output_path), index=False)

  if trace:
    print('Number of sub-hierarchies: {0}'.format(len(roots_idx)))
    print(df_subh.to_string())
    # print(df_subh)

  subh_pred_clss = [subh_pred_clss[roots_idx.index(x)] for x in df_subh.Root_idx]

  return df_subh.Root_idx.tolist(), subh_pred_clss


def neighborhood_information(adj, node_by_cl, nodes, cl_idx, anc_idx):
  """
  Extract the information of the association of a class and its ancestor in
  the neigborhood of the nodes in the graph (using the adjacency matrix)

  :param adj: Adjacency matrix of the nodes
  :type adj: np.matrix[float] *MxM*
  :param node_by_cl: Association matrix between nodes and classes
  :type node_by_cl: np.matrix[float] *MxN*
  :param nodes: List of indexes of nodes to be considered
  :type nodes: np.array[int]
  :param cl_idx: Index of the class to be analyzed
  :type cl_idx: int
  :param anc_idx: Index of the ancestor of the class to be analyzed
  :type anc_idx: int

  :return: Lis of prability of association between the 'nodes' and the class,
    the 'nodes' and its ancestor, and the 'nodes' and the class given the
    probability of association with its ancestor
  :rtype: np.array[np.array[float], np.array[float], np.array[float]]
  """
  ans = list(), list(), list()

  for node_idx in nodes:
    neighbors = np.nonzero(adj[:,node_idx])[0]
    ntotal = len(neighbors)
    nassc = np.count_nonzero(node_by_cl[neighbors,cl_idx])
    nance = np.count_nonzero(node_by_cl[neighbors,anc_idx])

    # proba. of being associated to class from neigh.
    ans[0].append(nassc/ntotal)
    # proba. of being associated to class ancestor from neigh.
    ans[1].append(nance/ntotal)
    # proba. of being associated to class given the number of neig assoc to its ancestor
    ans[2].append(nassc/nance if nance > 0 else 0)

  return ans


def create_txt(data, output_path, filename):
  """
  Create a txt file from a list of objects

  :param data: List of objects (strings or numbers) to be stored as txt file
  :type data: list[object]
  :param output_path: Path where the txt file is stored
  :type output_path: string
  :param filename: Name of the txt file
  :type filename: string
  """
  file = open('{0}/{1}.txt'.format(output_path, filename), 'w')
  file.write('\n'.join([str(x) for x in data]))
  file.close()


def hierarchy_to_tree(adj, cl_adj, node_by_cl, C, hier_cl_idx, output_path):
  """
  Generates the tree representation of the hierarchy from the adjacency and
  association matrices of graph and classes. Generates two files: the order of
  classes in the hierarchy (tree representation) and the ancestors of each class
  (in the same order)

  :param adj: Adjacency matrix of the nodes
  :type adj: np.matrix[float] *MxM*
  :param cl_adj: Adjacency matrix of the classes
  :type cl_adj: np.matrix[float] *NxN*
  :param node_by_cl: Association matrix between nodes and classes
  :type node_by_cl: np.matrix[float] *MxN*
  :param C: List of classes in the same order of the 'cl_adj' matrix
  :type C: np.array[string] *N*
  :param hier_cl_idx: list of classes (indexes) within the hiearchy (the first
    element of the list is the root class), where 'hier_cl_idx' is a subset
    of 'C'
  :type hier_cl_idx: np.array[int]
  :param output_path: Path where the output is stored
  :type output_path: string

  :return: Adjacency matrix of the corresponding subgraph of the hieararchy, i.e.,
    subgraph of all nodes associated to the root class of the hierarchy
  :rtype: np.matrix[float]
  """

  root_idx = hier_cl_idx[0]
  hier_nd_idx = np.nonzero(node_by_cl[:,root_idx])[0]
  hier_cl = C[hier_cl_idx] # terms to predict in hierarchy

  # Conver DAG to tree, will be used for prediction
  tree = mst(hier_nd_idx, hier_cl_idx, node_by_cl.copy(), cl_adj.copy())
  hier_cl_adj = np.zeros((len(hier_cl_idx),len(hier_cl_idx)))
  for i, idx in enumerate(hier_cl_idx):
    parents = direct_pa(idx, hier_cl_idx, tree)
    parents = [np.where(hier_cl_idx == p)[0][0] for p in parents]
    hier_cl_adj[i, parents] = 1

  # BFS hierarchy traverse
  queue = deque()
  queue.append((0,None,0))
  order = list() # list of classes in order of traverse
  ance_list = list() # list of ancestors of the classes in order of traverse

  while len(queue) != 0:
    pos, ance, d = queue.popleft()

    # Traverse according to BFS
    if d > 0: # root of hierarchy is not use for prediction
      order.append(hier_cl[pos])

      df = pd.DataFrame()
      df[hier_cl[pos]] = node_by_cl[hier_nd_idx, hier_cl_idx[pos]]

      if d > 1: # if the class has ancestor, root nod included as ancestor
        ance_list.append(hier_cl[ance])
        prb_feat = neighborhood_information(adj, node_by_cl, hier_nd_idx, hier_cl_idx[pos], hier_cl_idx[ance])
        df['prb1'] = prb_feat[0]
        df['prb2'] = prb_feat[1]
        df['prb3'] = prb_feat[2]
      else:
        ance_list.append('')

      # save data for the class in pos
      filename = re.sub(r'\W+', '', hier_cl[pos])
      df.to_csv('{0}/{1}.csv'.format(output_path, filename), index=False)

    for idx in np.nonzero(hier_cl_adj[:,pos])[0]:
      queue.append((idx, pos, d+1))

  create_txt(order, output_path, 'hierarchy')
  create_txt(ance_list, output_path, 'ancestors')

  return adj[np.ix_(hier_nd_idx,hier_nd_idx)].copy()


# Scale data
def scale_data(data):
  """
  Scale the data of a dataset without modifying the distribution of data.

  :param data: Dataset
  :type data: DataFrame

  :return: Dataset with scaled features
  :rtype: Dataframe
  """
  # MinMaxScaler does not modify the distribution of data
  minmax_scaler = MinMaxScaler() # Must be first option
  rob_scaler = RobustScaler() # RobustScaler is less prone to outliers

  new_data = pd.DataFrame()
  for fn in data.columns:
    scaled_feature = minmax_scaler.fit_transform(data[fn].values.reshape(-1,1))
    new_data[fn] = scaled_feature[:,0].tolist()

  return new_data


# compute structural properties and feature embedding
def compute_strc_prop(adj_mad, dimensions=16, p=1, q=0.5,
                      path=None, log=False, seed=None):
  """
  Compute multiple structural properties of the input network. Two types of
  properties are computed: hand-crafted and node embeddings.

  :param adj_mad: Adjacency matrix representation of the network, square and
    symmetric matrix.
  :type adj_mad: np.matrix[int]
  :param dimensions: Dimension of the node embedding, defaults to 16
  :type dimensions: int
  :param p: Return parameter of node2vec, defaults to 1
  :type p: float
  :param q: In-out parameter of node2vec, defaults to 0.5
  :type q: floar
  :param path: Relative path where the dataset will be saved, defaults to
    current path
  :type path: string
  :param log: Flag for logging of the results of the test, default to False
  :type log: bool
  :param seed: Random number seed, defaults to None
  :type seed: float

  :return: Dataset with scaled features representing the structural properties
    of the network and list of labels (names) of the features.
  :rtype: Tuple(Dataframe, List[string])
  """
  # create graph for adjacency matrix
  g = nx.Graph()
  edgelist = np.transpose(np.nonzero(adj_mad)).tolist()
  g.add_nodes_from(np.arange(len(adj_mad)))
  g.add_edges_from(edgelist)
  if log: print(nx.info(g))

  # node embedding for prediction
  workers = multiprocessing.cpu_count() // 2
  node2vec = Node2Vec(g, dimensions=dimensions, walk_length=5, num_walks=300, workers=workers, p=p, q=q)
  model = node2vec.fit(window=5, min_count=1, batch_words=5, workers=workers)
  embeddings = np.array([model.wv.get_vector(str(x)) for x in list(g.nodes)])

  # dimensionality reduction for clustering
  tsne = TSNE(n_components=2, random_state=seed, perplexity=15)
  embeddings_2d = tsne.fit_transform(embeddings)

  clustering_model = AffinityPropagation(damping=0.9)
  clustering_model.fit(embeddings_2d)
  yhat = clustering_model.predict(embeddings_2d)

  # igraph
  g = ig.Graph.Adjacency((adj_mad > 0).tolist())
  g = g.as_undirected()
  if log: prin(ig.summary(g))

  # get node properties form graph
  clust = np.array(g.transitivity_local_undirected(mode="zero"))
  deg = np.array(g.degree())
  neigh_deg = np.array(g.knn()[0])
  centr_betw = np.array(g.betweenness(directed=False))
  centr_clos = np.array(g.closeness())
  eccec = np.array(g.eccentricity())
  pager = np.array(g.personalized_pagerank(directed=False))
  const = np.array(g.constraint())
  hubs = np.array(g.hub_score())
  auths = np.array(g.authority_score())
  coren = np.array(g.coreness())
  diver = np.array(g.diversity())

  # add node properties to df
  # cretae dataset
  strc_df = pd.DataFrame()

  strc_df['clust'] = pd.Series(clust) # clustering
  strc_df['deg'] = pd.Series(deg) # degree
  strc_df['neigh_deg'] = pd.Series(neigh_deg) # average_neighbor_degree
  strc_df['betw'] = pd.Series(centr_betw) # betweenness_centrality
  strc_df['clos'] = pd.Series(centr_clos) # closeness_centrality
  strc_df['eccec'] = pd.Series(eccec) # eccentricity
  strc_df['pager'] = pd.Series(pager) # page rank
  strc_df['const'] = pd.Series(const) # constraint
  strc_df['hubs'] = pd.Series(hubs) # hub score
  strc_df['auths'] = pd.Series(auths) # authority score
  strc_df['coren'] = pd.Series(coren) # coreness
  strc_df['diver'] = pd.Series(diver) # diversity

  for i in range(dimensions):
    strc_df['emb_{0}'.format(i)] = pd.Series(embeddings[:,i])
  strc_df['emb_clust'] = pd.Series(yhat)

  columns = list(strc_df.columns)
  strc_df = scale_data(strc_df)
  if path != None:
    strc_df.to_csv('{0}/data.csv'.format(path), index=False)
  else:
    strc_df.to_csv('data.csv', index=False)

  return strc_df, columns
