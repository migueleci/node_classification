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


def plot_rocs(fprl, tprl, aucl, labels, filename, path):
  """
  Plot multiple roc curves in the same figure and save it in a PDF file

  :param fprl: Array of arrays of false positive rate values for multiple predictions
  :type fprl: np.arry[np.array[float]]
  :param tprl: Array of arrays of true positive rate values for multiple predictions
  :type tprl: np.arry[np.array[float]]
  :param aucl: Array of area under roc curve values for multiple predictions
  :type aucl: np.array[float]
  :param labels: Labels of the multiple models plotted
  :type labels: List[string]
  :param filename: Name of the PDF file
  :type filename: string
  :param path: Path where the plot will be stored.
  :type path: string
  """
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  for fpr, tpr, auc, l in zip(fprl, tprl, aucl, labels):
    plt.plot(fpr, tpr, lw=2, label='{0} = {1:.2f}'.format(l, auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('1 - Specificity')
  plt.ylabel('Sensitivity')
  plt.legend(loc='lower right')
  figname = '{0}/{1}_roc'.format(path, filename)
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


def plot_prs(recl, prcl, apl, labels, filename, path):
  """
  Plot multiple precision-recall curves in the same figure and save it in a PDF file

  :param recl: Array of arrays of recall values for multiple predictions
  :type recl: np.arry[np.array[float]]
  :param prcl: Array of arrays of precision values for multiple predictions
  :type prcl: np.arry[np.array[float]]
  :param apl: Array of average precision values for multiple predictions
  :type apl: np.array[float]
  :param labels: Labels of the multiple models plotted
  :type labels: List[string]
  :param filename: Name of the PDF file
  :type filename: string
  :param path: Path where the plot will be stored.
  :type path: string
  """
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  for rec, prc, ap, l in zip(recl, prcl, apl, labels):
    plt.plot(rec, prc, lw=2, label='{0} AP = {1:.2f}'.format(l, ap))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  figname = '{0}/{1}_ap'.format(path, filename)
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()
