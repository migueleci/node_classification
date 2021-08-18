Welcome to Node Classification
==============================

* Written by Miguel Romero
* Last update: 18/08/21

Node classification
-------------------

This package aims to provide different approaches to the *node classification*
problem (also known as *attribute prediction*) using machine learning
techniques. There are two approaches available: flat node classification (fnc)
and hierarchical classification (hc). Both approaches are based on a gradient
boosting decision tree algorithm called `XGBoost
<https://xgboost.readthedocs.io/en/latest/>`_, in addition the approaches are
equipped with an over-sampling technique call `SMOTE
<https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html>`_.

Flat classification
^^^^^^^^^^^^^^^^^^^

Flat node classification aims to valuate whether the structural (topological)
properties of a network are useful for predicting node attributes of
nodes (i.e., node classification), without considering the (possible)
relationships between the classes of the node attribute to be predicted, i.e.,
the classes are predicted independently.

Hierarchical classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hierarchical node classification considers the hierarchical organization of
the classes of a node attribute to be predicted. Using a top-down approach a
binary classifier is trained per class according to the hierarchy, which is
represented as a DAG.

Installation
------------

The node classification package can be install using pip, the requirements
will be automatically installed::

  python3 -m pip install nodeclass

The source code and examples can be found in this
`GitHub repository <https://github.com/migueleci/nodeclass>`_.

Example
-------

Flat classification
^^^^^^^^^^^^^^^^^^^

This example illustrates how the node classification package can be used
to check whether the structural properties of the gene co-expression network
improve the performance of the prediction of gene functions for rice
(*Oryza sativa Japonica*). In this example, a gene co-expression network
gathered from `ATTED II <https://atted.jp/>`_ is used.

How to run the example?
"""""""""""""""""""""""

The complete source code of the example can be found in the
`GitHub repository <https://github.com/migueleci/nodeclass>`_. First, the
*xgbfnc* package need to be imported::

  from nodeclass.models import xgbfn
  from nodeclass.tools import data

After creating adjacency matrix ``adj`` for the network, the structural
properties are computed using the module `data` of the package::

  df, strc_cols = data.compute_strc_prop(adj)

This method returns a DataFrame with the structural properties of the network
and a list of the names of these properties (i.e., column names). After adding
the additional features of the network to the DataFrame, the XGBhc module is
used to instantiate the XGBhc class::

  test = XGBhc()
  test.load_data(df, strc_cols, y, term, output_path='output')
  ans, pred, params = test.structural_test()

The data of the network is loaded using the ``load_data`` method. And the
structural test is execute using the ``structural_test`` method. The test
returns a boolean value which indicates whether the structural properties
help to improve the prediction performance, the prediction for the model
including the structural properties and its best parameters.

To run the example execute the following commands::

  cd test/flat_classification
  python3 test_small.py

Hierarchical classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example illustrates how the hierarchical classification package can
be used to predict gene functions considering the hierachical structure of
gene functions (as determined by `Gene Ontology <http://geneontology.org/>`_)
based on the gene co-expression network. This example uses the data for rice
(*Oryza sativa Japonica*),the gene co-expression network (GCN) was
gathered from `ATTED II <https://atted.jp/>`_.

How to run the example?
"""""""""""""""""""""""

The complete source code of the example can be found in the
`GitHub repository <https://github.com/migueleci/nodeclass>`_. First, the
*xgbhc* package need to be imported::

  from nodeclass.models import xgbhc
  from nodeclass.tools import data

The adjacency matrix for the GCN and the gene functions (from ancestral
relations of biological processes), and the matrix of associations between
genes and functions are created using the packaga ``data`` as follows::

  gcn, go_by_go, gene_by_go, G, T = data.create_matrices(data_ppi, data_isa, data_term_def, data_gene_term, OUTPUT_PATH, True)

The tree representation of the hierarchy is generated from the adjacency
matrix of the classes by removing the isolated classes, filtering the classes
according to the number of nodes associated (if required) and finding the
sub-hierarchies remaining. Then a
`minimum spanning tree <https://en.wikipedia.org/wiki/Minimum_spanning_tree>`_
(MST) algorithm is applied to each sub-hierarchy to get the its tree
representation (the order and ancestors of the classes will be calculated)::

  roots, subh_go_list = data.generate_hierarchy(gcn, go_by_go, gene_by_go, data_term_def, G, T, OUTPUT_PATH, filter=[5,300], trace=True)
  root, subh_go = roots[13], subh_go_list[13]
  subh_adj = data.hierarchy_to_tree(gcn, go_by_go, gene_by_go, T, subh_go, OUTPUT_PATH)

Additionally, the structural properties of the sub-graph of the GCN,
corresponding to the set of nodes associated to the classes in the
sub-hierarchy, are computed using the module `data`::

  data.compute_strc_prop(subh_adj, path=OUTPUT_PATH)

Finally, the XGBhc class is instantiated, the data of the sub-hierarchy is
loaded and the prediction is done as follows::

  model = XGBhc()
  model.load_data(data, root, hierarchy, ancestors, DATA_PATH, OUTPUT_PATH)
  model.train_hierarchy()

The results of the prediction are saved on the ``OUTPUT_PATH``, including the
roc and precision-recall curve, the confusion matrix and a csv file with some
performance metrics (such as the auc roc, average precision, recall, precision
and F1, true positive and true negative rate and the execution time).

To run the example execute the following commands::

  cd test/hierarchical_classification
  python3 test_data.py
  python3 test.py

Documentation
=============

..
  Documentation of the package can be found `here <https://xgbhc.readthedocs.io/en/latest/>`_.
