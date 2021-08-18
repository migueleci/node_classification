#!/usr/bin/python3
# coding: utf-8

# Hierarchical node classification (attribute prediction considering hierarchy of classes)
# Miguel Romero, 2021 ago 12

"""
Module for hierarchical node classification using a top-down approach.
"""

import os
import re
import sys
import datetime
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from time import time
from scipy import stats

# Own Libraries
from ..tools.plots import *

# Metrics
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# Cross-validation and scaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# Over-sampling and classifier Libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore',category=DeprecationWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)


class XGBhc:

  """
  Class for hierarchical node classification. This class builds an
  XGBoost binary classifier for each class following a top-down approach.

  :ivar data: Dataset with the graph information (topological properties of
    nodes) similar for all classes in the hierarchy.
  :vartype data: DataFrame
  :ivar hierarchy: List of classes in the hierarchy
  :vartype hierarchy: list[int]
  :ivar ancestors: List of ancestors of each class in the hierarchy, i.e., tree
    representation. Labels in 'ancestors'
    must match with the lebels of the classes in 'hierarchy'.
  :vartype ancestors: list[int]
  :ivar label: Label of the hierarchy, used to name the figures and output files.
  :vartype label: string
  :ivar data_path: Path where the data of the model is stored. In particular,
    the specific data for each class in the hierarchy. Files in 'data_path'
    must match with the lebels of the classes in 'hierarchy'.
  :vartype data_path: string
  :ivar output_path: Path where the output of the algorithm will be stored.
  :vartype output_path: string
  :ivar figs_path: Path where the figures will be stored.
  :vartype figs_path: string
  """

  def __init__(self):
    self.data = None
    self.label = None
    self.hierarchy = None
    self.ancestors = None
    self.data_path = None
    self.output_path = None
    self.figs_path = None


  def check_data(self):
    """
    Verify the input data.

    :return: flag indicating whether the input data is ok or not.
    :rtype: boolean
    """
    flag = True
    if flag and self.data is None:
      flag = False
      print("Variable 'data' must contain the dataset shared for all classes in hierarchy")
    elif flag and self.label is None:
      flag = False
      print("Variable 'label' must contain the label of the hierarchy for the outcomes of the algorithm")
    elif flag and self.hierarchy is None:
      flag = False
      print("Variable 'hierarchy' must contain the labels of the classes in the hierarchy")
    elif flag and self.ancestors is None:
      flag = False
      print("Variable 'ancestors' must contain the labels of the ancestors for each class in the hierarchy")
    elif flag and len(self.hierarchy) != len(self.ancestors):
      flag = False
      print("Variable 'hierarchy' and 'ancestor' must be of the same size")
    elif flag and self.data_path is None:
      flag = False
      print("Variable 'data_path' must contain the path of the datasets specific for each class of the hierarchy")
    return flag


  def load_data(self, data, label, hierarchy, ancestors, data_path, output_path=None, figs_path=None):
    """
    Load the data of the network and the hierarchy of classes.

    :param data: Dataset with the graph information (topological properties of
      nodes) similar for all classes in the hierarchy.
    :type data: DataFrame
    :param hierarchy: List of classes in the hierarchy
    :type hierarchy: list[int]
    :param ancestors: List of ancestors of each class in the hierarchy, i.e., tree
      representation.
    :type ancestors: list[int]
    :param label: Label of the hierarchy, used to name the figures and output files.
    :type label: string
    :param data_path: Path where the data of the model is stored. In particular,
      the specific data for each class in the hierarchy.
    :type data_path: string
    :param output_path: Path where the output of the algorithm will be stored,
      defaults to `"YYYY-MM-DD/"`.
    :type output_path: string
    :param figs_path: Path where the figures will be stored, defaults
      to `"YYYY-MM-DD/"`.
    :type figs_path: string
    """
    self.data = data
    self.label = re.sub('[\W_]+', '', label)
    self.hierarchy = hierarchy
    self.ancestors = ancestors
    self.data_path = data_path
    if output_path is None:
      dt = datetime.datetime.today()
      self.output_path = "{2}-{1}-{0}".format(dt.day, dt.month, dt.year)
    else:
      self.output_path = output_path
    if figs_path is None: self.figs_path = self.output_path
    else: self.figs_path = figs_path
    self.create_path(self.output_path)
    self.create_path(self.figs_path)
    self.check_data()


  def create_path(self, path):
    """
    Create a path.

    :param path: Relative path to be created.
    :type path: string
    :raises OSError: the path already exist
    """
    try:
      os.makedirs(path, exist_ok=True)
    except Exception as e:
      print('Something is wrong with the output path: {0}'.format(self.output_path))
      print('Verify the path and try again. {0}.'.format(e))


  def create_classifier(self, n_iter=5, n_jobs_cv=None, n_jobs_xgb=2,
                      eval_metric="aucpr", scoring="recall", seed=None):
    """
    Builds the binary classifier within a hyper-parameters tuning model.

    :param n_iter: Number of iterations in cross validation for
      hyper-parameters tuning, defaults to 5
    :type n_iter: int
    :param n_jobs_cv: Number of parallel jobs running for hyper-parameters
      tuning, defaults to None
    :type n_jobs_cv: int
    :param n_jobs_xgb: Number of parallel jobs running for training the
      classifier, defaults to 2
    :type n_jobs_xgb: int
    :param eval_metric: Evaluation metric for training the classifier,
      defaults to "aucpr"
    :type eval_metric: string
    :param scoring: Scoring metric for hyper-parameters tuning, defaults
      to "recall"
    :type scoring: string
    :param seed: Random number seed, defaults to None
    :type seed: int

    :return: Hyper-parameter tuning model with XGBoost binary classsifier
    :rtype: RandomizedSearchCV
    """

    if n_jobs_cv is None: n_jobs = multiprocessing.cpu_count() // 2

    _param_grid = {
      'max_depth': [6, 10, 15, 20],
      'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
      'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
      'gamma': [0, 0.25, 0.5, 1.0],
      'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

    param_grid = {
      'max_depth': [3, 6, 10],
      'min_child_weight': [0.5, 3.0, 5.0, 8.0],
      'eta': [0.01, 0.05, 0.2, 0.4],
      'subsample': [0.5, 0.7, 0.9, 1.0]}

    clf = xgb.XGBClassifier(booster='gbtree', n_jobs=n_jobs_xgb,
                            eval_metric=eval_metric, random_state=seed)

    return RandomizedSearchCV(clf, param_grid, n_iter=n_iter, n_jobs=n_jobs_cv,
                              scoring=scoring, random_state=seed)


  def opt_threshold(self, y_orig, y_pred):
    """
    Compute the classification from probabilities based on the optimum
    threshold according to precision-recall curve, that is the threshold
    that maximies the F1 score.

    :param y_orig: Truth values of the prediction.
    :type y_orig: np.array[int]
    :param y_pred: Predicted probabilities with the XGBoost classifier.
    :type y_pred: np.array[float]

    :return: Classification for the input array of probabilities which
      maximies F1 score.
    :rtype: np.array[int]
    """
    prec, recall, thresh = precision_recall_curve(y_orig, y_pred)
    fscore = (2*prec*recall)/(prec+recall)
    ix = np.argmax(fscore)
    opt_thr = thresh[ix]
    ynew = y_pred.copy()
    ynew[ynew >= opt_thr] = 1
    ynew[ynew < opt_thr] = 0
    return ynew


  def evaluate(self, y_orig, y_pred_prob):
    """
    Evaluate the performance of the prediction using metrics such as the
    auc roc, average precision score, precision, recall and F1 score.

    :param y_orig: Truth values of the prediction.
    :type y_orig: np.array[int]
    :param y_pred_prob: Predicted probabilities with the XGBoost classifier.
    :type y_pred_prob: np.array[float]

    :return: Evaluation metrics for the prediction.
    :rtype: dict[string->float]
    """
    roc = roc_auc_score(y_orig, y_pred_prob) # ROC AUC
    ap = average_precision_score(y_orig, y_pred_prob) # Average precision

    y_pred_new = self.opt_threshold(y_orig, y_pred_prob)
    ncm = confusion_matrix(y_orig, y_pred_new, normalize='true') # normalized confusion matrix
    f1s = f1_score(y_orig, y_pred_new) # f1 score
    prec, recall, fscore, support = precision_recall_fscore_support(y_orig, y_pred_new) # precision and recall

    mts = {'roc':roc,'ap':ap,'f1s':f1s,'prec':prec[1],
           'rec':recall[1],'tnr':ncm[0,0],'tpr':ncm[1,1]}

    return mts, y_pred_new


  def plot_performance(self, yorig, ypred):
    """
    Plot roc curve, precision-recall curve and confussion matrices for a
    prediction.

    :param y_orig: Truth values of the prediction.
    :type y_orig: np.array[int]
    :param y_pred_prob: Predicted probabilities with the XGBoost classifier.
    :type y_pred_prob: np.array[float]
    """
    # ROC AUC
    roc = roc_auc_score(yorig, ypred)
    fpr, tpr, _ = roc_curve(yorig, ypred)
    plot_roc(fpr, tpr, roc, '{0}'.format(self.label), self.figs_path)

    # Average precision
    ap = average_precision_score(yorig, ypred)
    prec, recall, thresh = precision_recall_curve(yorig, ypred)
    plot_pr(recall, prec, ap, '{0}'.format(self.label), self.figs_path)

    # compute best threshold, according to PR curve
    pred_new = self.opt_threshold(yorig, ypred)
    cm = confusion_matrix(yorig, pred_new, normalize='true') # normalized confusion matrix
    plot_conf_matrix(cm, '{0}'.format(self.label), self.figs_path)


  def write_csv(self, pred):
    """
    Save the evaluation metrics for prediction of both models, i.e., without
    and with structural properties.

    :param pred: Predicted probabilities for the model.
    :type pred: np.array[float]
    """
    df_resume = pd.DataFrame()
    metrics = pred.keys()
    df_resume['Metric'] = pd.Series(metrics)
    df_resume['Score'] = pd.Series([pred[k] for k in metrics])
    df_resume.to_csv('{0}/{1}_resume.csv'.format(self.output_path, self.label), index=False)


  def train_class(self, X, y, label, n_splits, seed, n_iter=5, n_jobs_cv=None,
            n_jobs_xgb=2, eval_metric="aucpr", scoring="recall"):
    """
    Training using a combination of SMOTE (Over-Sampling) and XGBoost techniques.

    :param X: X
    :type X: DataFrame
    :param y: y
    :type y: np.array
    :param n_splits: n_splits
    :type n_splits: int
    :param seed: seed
    :type seed: int
    :param n_iter: Number of iterations in cross validation for
      hyper-parameters tuning, defaults to 5
    :type n_iter: int, optional
    :param n_jobs_cv: Number of parallel jobs running for hyper-parameters
      tuning, defaults to None
    :type n_jobs_cv: int
    :param n_jobs_xgb: Number of parallel jobs running for training the
      classifier, defaults to 2
    :type n_jobs_xgb: int
    :param eval_metric: Evaluation metric for training the classifier,
      defaults to "aucpr"
    :type eval_metric: string
    :param scoring: Scoring metric for hyper-parameters tuning, defaults to
      "recall"
    :type scoring: string

    :return: Predicted probabilities with the XGBoost classifier, feature
      importance measured by total gain, and best parameter combination for
      the classfifier.
    :rtype: Tuple(np.array[float], dict[string->float], dict[string->float])
    """

    y_pred_prob = np.zeros(len(y))
    fimp = dict([(c,0.0) for c in X.columns])
    params = dict()

    # Implementing SMOTE Technique, Cross Validating the right way
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in sss.split(X, y):
      Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
      ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

      Xtrain, Xtest = Xtrain.values, Xtest.values # Turn into an array
      ytrain, ytest = ytrain.values, ytest.values

      if np.sum(ytrain) == 0 or np.sum(ytrain) == len(ytrain): continue
      if np.sum(ytest) == 0 or np.sum(ytest) == len(ytest): continue

      rand_xgb = self.create_classifier(n_iter=n_iter, n_jobs_cv=n_jobs_cv, n_jobs_xgb=n_jobs_xgb,
                          eval_metric=eval_metric, scoring=scoring, seed=seed)
      try:
        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_xgb)
        model = pipeline.fit(Xtrain, ytrain)
      except:
        pipeline = imbalanced_make_pipeline(RandomOverSampler(sampling_strategy='minority'), rand_xgb)
        model = pipeline.fit(Xtrain, ytrain)

      best_est = rand_xgb.best_estimator_
      pred_proba = best_est.predict_proba(Xtest)[:,1]
      y_pred_prob[test_index] = pred_proba

      total_gain = np.sum(best_est.feature_importances_)
      feature_importances_ = best_est.feature_importances_ / total_gain
      for col, val in zip(X.columns, feature_importances_): # default importance type
        if not np.isnan(val):
          fimp[col] += val

      for p in rand_xgb.best_params_:
        if p not in params: params[p]=list()
        params[p]+=[rand_xgb.best_params_[p]]

    for p in params:
      mode = stats.mode(params[p])
      params[p] = mode[0][0] if mode[1][0] > 1 else "Any"

    return y_pred_prob, fimp, params


  def train_hierarchy(self, n_splits=5, seed=None):
    """
    Hierarchical classification of nodes using a local classifier. This approach uses
    a bfs to traverse the hierarchy, represented as a tree (no node has more than ones
    parent).

    :param n_splits: n_splits
    :type n_splits: int
    :param seed: seed
    :type seed: int

    :return: Predicted probabilities and classification predicted with the
      algorithm, feature importance measured by total gain, and best parameter
      combination for the classfifier.
    :rtype: Tuple(np.array[float], dict[string->float], dict[string->float])
    """

    pred_prob, pred, performance = None, None, None

    if self.check_data():

      s = time()
      y_orig = np.zeros((len(self.data), len(self.hierarchy)))

      ###
      # Hierarchy prediction
      ###
      pred = np.zeros((len(self.data), len(self.hierarchy)))
      pred_prob = np.zeros((len(self.data), len(self.hierarchy)))
      fimp_hier = dict([(x, 0.0) for x in self.data.columns])

      for idx, (node, ance) in tqdm(enumerate(zip(self.hierarchy, self.ancestors)), desc="Training", total=len(self.hierarchy)):
        X = pd.read_csv('{0}/{1}.csv'.format(self.data_path, node.replace(":","")), dtype='float')
        y = X[node]
        X = pd.concat([self.data, X], axis=1).drop([node], axis=1)
        if len(ance) > 0:
          X[ance] = pred_prob[:,self.hierarchy.index(ance)]

        y_orig[:,idx] = y
        pred_prob[:,idx], fimp, params = self.train_class(X, y, node, n_splits, seed)
        for k in fimp:
          if k not in fimp_hier: fimp_hier[k] = 0.0
          fimp_hier[k] += fimp[k]

        # fix inconsistencies (true-path rule)
        if len(ance) > 0:
          ance_idx = self.hierarchy.index(ance)
          pred_prob[:,idx] = pred_prob[:,idx] * pred_prob[:,ance_idx]

      ###
      # Evaluation of hierarchy
      ###
      y_orig = y_orig.flatten()
      pred_prob = pred_prob.flatten()

      # compute performance metrics for trial
      performance, pred = self.evaluate(y_orig, pred_prob)
      f = time()
      performance['time'] = f-s
      performance['nodes'] = len(self.data)
      performance['classes'] = len(self.hierarchy)

      self.plot_performance(y_orig, pred_prob)
      self.write_csv(performance)

    else:
      print('Data must be loaded to start the analysis. Use the load_data() method.')

    return pred_prob, pred, performance, fimp
