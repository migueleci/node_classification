#!/usr/bin/python3
# coding: utf-8

# Hierarchical node classification (attribute prediction considering hierarchy of classes)
# MST algorithm
# Miguel Romero, 2021 ago 12

"""
Module containing minimum spanning tree algorithm (MST) to turn a hierarchy of
classes represented as a directed acyclic graph (DAG) into a tree. The
algorithm uses the number of nodes associated to each class to select to select
a parent class.

This algorithm is an adaptation of the algorithm used in (`Jiang et. al. 2008
<http://www.biomedcentral.com/1471-2105/9/350>`_).
"""

import scipy.stats
import numpy as np
from tqdm import tqdm


def direct_pa(cl, cls, hie):
  """
  Find the direct parents for a given class in a given heirarchy. The direct
  parents of a class are the set of classes that do not have any other
  descendant.

  :param cl: List of indexes of the class for which the direct parents are
    being searched
  :type cl: string
  :param cls: List of indexes of the classes in the hierarchy to be considered
    by the algorithm
  :type cls: list[string] of size *N*
  :param hie: Adjacency matrix representing the hierarchy of classes considered
    in 'cls'
  :type hie: np.matrix[float] of size *NxN*

  :return: List of indexes of direct parents of the class 'cl' in the hierarchy
  :rtype: list[int]
  """
  cl_id = np.nonzero(cls == cl)[0]

  cand_pa_id = np.nonzero(hie[cl_id,:])[1] # Find all candidate parent cls of term cl
  sub_hie_idx = np.hstack((cand_pa_id, cl_id))
  sub_hie = hie[np.ix_(sub_hie_idx, sub_hie_idx)]

  for i in range(len(cand_pa_id)):
    # If the number of descendants of candidate parent term i is greater than 1, it is NOT the
    # direct parent term. Discard all cls with more than 1 descendant
    if np.count_nonzero(sub_hie[:,i]) > 1:
      cand_pa_id[i] = -1 # Python index from 0, -1 means that is no the direct parent

  pa_id = cand_pa_id[cand_pa_id >= 0] # any value diff from -1 is related as a direct parent
  return cls[pa_id] # return id of parent(s) for term cl


def mst(nodes, cls, node_by_cl, cl_by_cl):
  """
  Minimal spanning tree (MST) algorithm for a hierarchy of classes.

  :param nodes: List of indexes of the nodes to be considered by the algorithm
  :type nodes: np.arrat[int] of size *M*
  :param cls: List of indexes of the classes in the hierarchy to be considered
    by the algorithm
  :type cls: list[string] of size *N*
  :param node_by_cl: Matrix representing the associations between nodes and
    classes, where *P\geqN* is the total number of classes in the hierarchy.
  :type node_by_cl: np.matrix(float) of size *PxM*
  :param cl_by_cl: Adjacency matrix representing the hierarchy of classes
  :type cl_by_cl: np.matrix(float) of size *PxP*

  :return: Adjacency matrix of the tree representation of the hierarchy of
    classes considered in 'cls'
  :rtype: np.matrix[float] of size *NxN*
  """
  hie = cl_by_cl[np.ix_(cls,cls)].copy()
  n = len(cls)

  # From the 2nd level (level directly below the root) and lower
  # discard cls with exactly one parent (thery are already a tree)
  leaf = cls[1:].copy()
  leaf[np.nonzero(hie.sum(axis=1) == 1)[0]-1] = -1
  leaf = leaf[leaf >= 0]

  if len(leaf) > 0: # If no 2nd level cls, keep the current hierarchy
    for i in range(len(leaf)):
      parents = direct_pa(leaf[i], cls, hie)
      if len(parents) > 1: # check if leaf has only one or multiple ancestors
        leaf_id = np.nonzero(cls == leaf[i])[0]
        p = np.zeros(len(parents))

        for j in range(len(parents)):
          a = node_by_cl[nodes, leaf[i]]
          b = node_by_cl[nodes, parents[j]]

          both, one = np.count_nonzero(a+b == 2), np.count_nonzero(a+b == 1)
          if both + one > 0:
            p[j] = both / (both + one)

        pa = parents[p == np.max(p)]
        if len(pa) > 1:
          pa = pa[np.random.randint(0, len(pa), 1)]
        pa_id = np.nonzero(cls == pa)[0]

        # Relabel the entries in hie
        hie[leaf_id,:] = 0
        ance_id1 = np.hstack((np.nonzero(hie[pa_id,:])[1], pa_id))
        hie[leaf_id, ance_id1] = 1
        ch_id = np.nonzero(hie[:, leaf_id])[0]
        if len(ch_id) > 0:
          hie[ch_id,:] = 0
          ance_id2 = np.hstack((np.nonzero(hie[leaf_id,:])[1], leaf_id))
          hie[np.ix_(ch_id, ance_id2)] = 1

  return hie
