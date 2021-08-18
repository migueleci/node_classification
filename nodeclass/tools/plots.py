#!/usr/bin/python3
# coding: utf-8

# Hierarchical node classification (attribute prediction considering hierarchy of classes)
# Plotting module
# Miguel Romero, 2021 ago 12

"""
Module for plotting the results of the prediction.
"""

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt

rc('font', family='serif', size=18)
rc('text', usetex=False)

# Default colors
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#7f7f7f', '#bcbd22', '#17becf']


def plot_roc(fpr, tpr, auc, filename, path):
  """
  Plot a ROC curve and save it in a PDF file

  :param fpr: Array of false positive rate values
  :type fpr: np.array[float]
  :param tpr: Array of true positive rate values
  :type tpr: np.array[float]
  :param auc: Area under roc curve
  :type auc: float
  :param filename: Name of the PDF file
  :type filename: string
  :param path: Path where the plot will be stored.
  :type path: string
  """
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(fpr, tpr, lw=2, label='AUC = {0:.2f}'.format(auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('1 - Specificity')
  plt.ylabel('Sensitivity')
  plt.legend(loc='lower right')
  figname = '{0}/{1}_roc.pdf'.format(path, filename)
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


def plot_pr(rec, prc, ap, filename, path):
  """
  Plot a precision-recall curve and save it in a PDF file

  :param rec: Array of recall values
  :type rec: np.array[float]
  :param prc: Array of precision values
  :type prc: np.array[float]
  :param ap: Average precision score
  :type ap: float
  :param filename: Name of the PDF file
  :type filename: string
  :param path: Path where the plot will be stored.
  :type path: string
  """
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(rec, prc, lw=2, label='AP = {0:.2f}'.format(ap))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  figname = '{0}/{1}_ap'.format(path, filename)
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


def plot_conf_matrix(cm, filename, path, labels=[0,1]):
  """
  Plot a confusion matrix and save it in a PDF file

  :param cm: Confusion matrix
  :type cm: np.matrix[float]
  :param filename: Name of the PDF file
  :type filename: string
  :param path: Path where the plot will be stored.
  :type path: string
  :param labels: Labels of the classes used in both axis of the matrix, default to [0,1].
  :type labels: List[float]
  """
  df_cm = pd.DataFrame(cm, index=labels, columns=labels)

  fig, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
  figname = '{0}/{1}_cm'.format(path, filename)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


# # plot multiple auc roc curves in one figure
# def plot_mroc(term, folder, fprs, tprs, aucs, lbs):
#   fig, ax = plt.subplots(figsize=(6.5,6.5))
#   for fpr, tpr, auc, lb in zip(fprs, tprs, aucs, lbs):
#     plt.plot(fpr, tpr, lw=2, label='{0} AUC = {1:.2f}'.format(lb, auc))
#   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#   plt.xlim([0.0, 1.0])
#   plt.ylim([0.0, 1.0])
#   plt.xlabel('1 - Specificity')
#   plt.ylabel('Sensitivity')
#   plt.legend(loc='lower right')
#   plt.savefig('{0}/{1}_auc.pdf'.format(folder, term), format='pdf', dpi=600)
#   plt.close()
#
#
# # plot multiple average precision curves in one figure
# def plot_map(term, folder, recalls, precisions, aps, lbs):
#   fig, ax = plt.subplots(figsize=(6.5,6.5))
#   for recall, precision, ap, lb in zip(recalls, precisions, aps, lbs):
#     plt.plot(recall, precision, lw=2, label='{0} AP = {1:.2f}'.format(lb, ap))
#   plt.xlim([0.0, 1.0])
#   plt.ylim([0.0, 1.0])
#   plt.xlabel('Recall')
#   plt.ylabel('Precision')
#   plt.legend(loc='lower right')
#   plt.savefig('{0}/{1}_ap.pdf'.format(folder, term), format='pdf', dpi=600)
#   plt.close()
#
# # plot line with std dev
# def plot_mts_hist(x,y,e,folder,name):
#   fig, ax = plt.subplots(figsize=(6.5,6.5))
#   plt.plot(x, y, '--')
#   lowerb = [yi-ei for yi,ei in zip(y,e)]
#   upperb = [yi+ei for yi,ei in zip(y,e)]
#   plt.fill_between(x, lowerb, upperb, alpha=.3)
#   plt.ylabel(name)
#   plt.xticks(rotation=90)
#   plt.tight_layout()
#   plt.savefig('{0}/{1}_hist.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.close()
#
# # plot pie
# def plot_feat_imp(l,x,folder,name):
#   fig, ax = plt.subplots(figsize=(8,5))
#   plt.pie(x, autopct='%1.1f%%', textprops=dict(size=14, color='w', weight='bold'))
#   plt.legend(l, loc='best')
#   plt.axis('equal')
#   plt.tight_layout()
#   plt.savefig('{0}/{1}_fimp.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.close()
#
#
# # plot auc roc vs height in hierarchy
# def plot_auc_height(data,folder,name):
#   x, y = [x for x,y in data], [y for x,y in data]
#   fig, ax = plt.subplots(figsize=(5,5))
#   plt.plot(x,y,'.')
#   plt.xlabel('Height')
#   plt.ylabel('AUC ROC')
#   plt.tight_layout()
#   plt.savefig('{0}/{1}_auc_height.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.close()
#
# # plot multiple line plots
# def line_plot(data,model_name,xlabel,ylabel,xticks,ylim,path):
#   fig, ax = plt.subplots(figsize=(8,8))
#   x = np.arange(len(xticks))
#   for idx, (_data, _model_name) in enumerate(zip(data,model_name)):
#     plt.plot(x, _data, 'b-', color=COLORS[idx], label=_model_name)
#   plt.ylim(ylim)
#   plt.xlabel(xlabel)
#   plt.ylabel(ylabel)
#   plt.xticks(x, xticks, rotation=90)
#   plt.legend(loc='best')
#   plt.tight_layout()
#   fname = re.sub(r'\W+', '', ylabel).lower()
#   plt.savefig('{0}/{1}.pdf'.format(path, fname), format='pdf', dpi=600)
#   plt.close()
