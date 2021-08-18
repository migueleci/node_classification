#!/usr/bin/python3
# coding: utf-8

# Hierarchical node classification (attribute prediction considering hierarchy of classes)
# Test of algorithm (data processing)
# Miguel Romero, 2021 ago 12

import numpy as np
import pandas as pd


# nodeclass library
from nodeclass.tools import data

PATH = "/home/miguel/projects/omics/XGBhc/test"
DATA_PATH = "{0}/{1}".format(PATH, "raw_data")
OUTPUT_PATH = "{0}/{1}".format(PATH, "data")

data_ppi = pd.read_csv('{0}/data_ppi.csv'.format(DATA_PATH), dtype='object', header=0, names=['Source','Target'])
data_isa = pd.read_csv('{0}/data_isa.csv'.format(DATA_PATH), dtype='object', header=0, names=['Class','Ancestor'])
data_term_def = pd.read_csv('{0}/data_term_def.csv'.format(DATA_PATH), dtype='object', header=0, names=['Class','Desc'])
data_gene_term = pd.read_csv('{0}/data_gene_term.csv'.format(DATA_PATH), dtype='object', header=0, names=['Node','Class'])

data_isa = data_isa[data_isa['Class']!='GO:0008150']
data_isa = data_isa[data_isa['Ancestor']!='GO:0008150']
data_term_def = data_term_def[data_term_def['Class']!='GO:0008150']
data_gene_term = data_gene_term[data_gene_term['Class']!='GO:0008150']

gcn, go_by_go, gene_by_go, G, T = data.create_matrices(data_ppi, data_isa, data_term_def, data_gene_term, OUTPUT_PATH, True)

roots, subh_go_list = data.generate_hierarchy(gcn, go_by_go, gene_by_go, data_term_def, G, T, OUTPUT_PATH, filter=[5,300], trace=True)

root, subh_go = roots[13], subh_go_list[13]
subh_adj = data.hierarchy_to_tree(gcn, go_by_go, gene_by_go, T, subh_go, OUTPUT_PATH)

data.compute_strc_prop(subh_adj, path=OUTPUT_PATH)
