#!/usr/bin/env python
# coding: utf-8

# Gene function prediction for Oryza sativa Japonica
# Miguel Romero 07/07/21

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

from xgbfnc import XGBfnc
from xgbfnc import data

##########################
# 1. Import and clean data
##########################

print('Loading Data...')
data_gcn = pd.read_csv('data/data_gcn.csv', dtype='object')
data_term_def = pd.read_csv('data/data_term_def.csv', dtype='object')
data_gene_term = pd.read_csv('data/data_gene_term.csv', dtype='object')

data_term_def = data_term_def[data_term_def['Term']!='GO:0008150']
data_gene_term = data_gene_term[data_gene_term['Term']!='GO:0008150']

# genes
# nG:number of genes, idxG:gene index map
G = np.array(sorted(list(set(data_gcn['Source'].tolist()+data_gcn['Target'].tolist()))))
G = np.random.choice(G, int(len(G)*0.1)) # Take 10% of the GCN
nG, idxG = len(G), dict([(p,i) for i,p in enumerate(G)])

# functions
# nF:number of functions, idxF:function index map
F = np.array(sorted(list(set(data_term_def['Term'].tolist()+data_gene_term['Term'].tolist()))))
nF, idxF = len(F), dict([(g,i) for i,g in enumerate(F)])


##############################
# 2. Create matrices from data
##############################

# GCM matrix
gcn = np.zeros((nG,nG))
for edge in tqdm([tuple(x) for x in data_gcn.to_numpy()]):
  if edge[0] in idxG and edge[1] in idxG:
    u, v = idxG[edge[0]], idxG[edge[1]]
    gcn[u][v] = gcn[v][u] = 1

# gene by go matrix
gene_by_go = np.zeros((nG,nF))
for edge in tqdm([tuple(x) for x in data_gene_term.to_numpy()]):
  if edge[0] in idxG:
    u, v = idxG[edge[0]], idxF[edge[1]]
    gene_by_go[u,v] = 1

print()
print('**Data**')
print('Genes: \t{0:6}'.format(len(gcn)))
print('Genes annot.: \t{0:6}'.format(np.count_nonzero(gene_by_go)))
print('Co-expressed: \t{0:6.0f}'.format(np.sum(gcn)/2))
print('Functions: \t{0:6}'.format(gene_by_go.shape[1]))


#####################################
# 3. Prepare term data for prediction
#####################################

print()
print('**Function filtering**')
# Prune terms according to paper, very specific and extremes with little to
# no information terms are avoided. Select genes used for prediction
# Accoding to restriction 10 <= genes annotated <= 300
terms_pred_idx = list()
for i in range(nF):
  count = np.count_nonzero(gene_by_go[:,i])
  if 10 <= count <= 300:
    terms_pred_idx.append(i)
print('Number of filtered functions: {0}'.format(len(terms_pred_idx)))


###################
# 4. Design dataset
###################

print()
print('Computing structural properties...')
df, strc_cols = data.compute_strc_prop(gcn)

# Select one function randomly for prediction
term_idx = np.random.choice(terms_pred_idx, 1)[0]
term = F[term_idx]

for idx, trm in zip(terms_pred_idx, F[terms_pred_idx]):
  df[trm] = pd.Series(gene_by_go[:,idx])
# df.to_csv('data_resume.csv', index=False)
y = df[term]
df = df.drop(columns=[term])

###############
# 5. Prediction
###############

print()
print('**Prediction**')
print('Term: {0}'.format(term))

test = XGBfnc()
test.load_data(df, strc_cols, y, term, output_path='output')
print()
print('Testing...')
ans, pred, params = test.structural_test()

print('')
print('Test result: structural properties {0} improve the prediction performance'.format('does' if ans else "doesn't"))
print('Training parameters: {0}'.format(params))
