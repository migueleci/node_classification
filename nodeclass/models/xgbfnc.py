#!/usr/bin/env python
# coding: utf-8

# Node classification (attribute prediction) - flat approach
# Miguel Romero, 2021 jul 1

"""
Module for flat node classification and testing the importance of the
structural properties of the network.
"""

# General libraries
import os
import datetime
import numpy as np
import multiprocessing
from scipy import stats

# # Plotting, own library
from ..tools.plots import *

# Classifier Libraries
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

# Cross-validation and scaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# Over-sampling and classifier Libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

# XGBoost classifier
import xgboost as xgb

# Other libraries
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


class XGBfnc:
  """
  Class for flat node classification. This class builds two XGBoost binary
  classifier for the attribute prediction using two datasets with different
  features, one including structural properties of the network and the other
  one without them.

  :ivar df: Datasets with all features of the network.
  :vartype df: DataFrame
  :ivar orig_cols: List of feature names (columns of df)
    **non** realted to structural properties of the network.
  :vartype orig_cols: List[string]
  :ivar strc_cols: List of feature names (columns of df)
    realted to structural properties of the network. The intersection between
    **orig_cols** and **strc_cols** must be empty.
  :vartype strc_cols: List[string]
  :ivar y: Serie representing the node attribute to be predicted.
  :vartype y: Series
  :ivar ylabel: Name of the node attribute to be predcited.
  :vartype ylabel: string
  :ivar output_path: Path where the output of the algorithm will
    be stored.
  :vartype output_path: string
  :ivar figs_pat: Path where the figures will be stored.
  :vartype figs_pat: string
  """

  # Default constructor
  def __init__(self):
    self.df = None # dataframe with node features
    self.orig_cols = None # list of features non related to structural properties
    self.strc_cols = None # list of features realted to structural properties
    self.y = None # feature to predict (binary class), column of df
    self.ylabel = None # feature to predict (binary class), column of df
    self.output_path = None # path for output of the model (csv)
    self.figs_path = None # path for plotting


  def load_data(self, df, strc_cols, y, ylabel, output_path=None, figs_path=None):
    """
    Load the data of the network.

    :param df: Dataset with all node features.
    :type df: DataFrame
    :param strc_cols: List of features related to structural properties.
    :type strc_cols: List[string]
    :param y: Node attribute to be predicted. Should be the same size as the df.
    :type y: Series
    :param ylabel: Name of the node attribute to be predicted.
    :type ylabel: string
    :param output_path: Path to save output, defaults to `"YYYY-MM-DD/"`.
    :type output_path: string
    :param figs_path: Path to save figs, defaults to `"YYYY-MM-DD/"`.
    :type figs_path: string

    """
    self.df = df # dataframe
    self.strc_cols = strc_cols
    self.orig_cols = [c for c in df.columns if c not in strc_cols]
    self.y = y
    self.ylabel = ylabel # label of attribute to predict

    if output_path is None:
      dt = datetime.datetime.today()
      self.output_path = "{2}-{1}-{0}".format(dt.day, dt.month, dt.year)
    else: self.output_path = output_path
    if figs_path is None: self.figs_path = self.output_path
    else: self.figs_path = figs_path

    self.create_path(self.output_path)
    self.create_path(self.figs_path)


  def create_path(self, path):
    """
    Create a path.

    :param path: Relative path to be created.
    :type path: string
    :raises OSError: the path already exist
    """
    try:
      os.makedirs(self.output_path, exist_ok=True)
    except Exception as e:
      print('Something is wrong with the output path: {0}.'.format(self.output_path))
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

    return mts


  # print prediction performance for test (cv) and validation
  def print_performance(self, scores, title):
    """
    Print the evaluation metrics for the prediction.

    :param scores: Evaluation metrics for the prediction.
    :type scores: dict[string->float]
    :param title: Name of the model or experiment.
    :type title: string
    """
    print('\n*** {0}'.format(title))
    print('AUC ROC: {0:.3f}'.format(scores['roc']))
    print('Average precision: {0:.3f}'.format(scores['ap']))
    print('F1 score: {0:.3f}'.format(scores['f1s']))
    print('Recall: {0:.3f}'.format(scores['rec']))
    print('Precision: {0:.3f}'.format(scores['prec']))
    print('True positive rate: {0:.3f}'.format(scores['tpr']))
    print('True negative rate: {0:.3f}'.format(scores['tnr']))


  def plot_performance(self, a, label):
    """
    Plot roc curve, precision-recall curve and confussion matrices for a
    prediction.

    :param a: Predicted probabilities for the model.
    :type a: np.array[float]
    :param label: Label of the models for the plots.
    :type label: string
    """
    # ROC AUC
    roc_a = roc_auc_score(self.y, a)
    fpr_a, tpr_a, _ = roc_curve(self.y, a)
    plot_roc(fpr_a, tpr_a, roc_a, '{0}_{1}'.format(self.ylabel, label), self.figs_path)

    # Average precision
    ap_a = average_precision_score(self.y, a)
    prec_a, recall_a, thresh_a = precision_recall_curve(self.y, a)
    plot_pr(recall_a, prec_a, ap_a, '{0}_{1}'.format(self.ylabel, label), self.figs_path)

    # compute best threshold, according to PR curve
    anew = self.opt_threshold(self.y, a)
    ancm = confusion_matrix(self.y, anew, normalize='true') # normalized confusion matrix
    plot_conf_matrix(ancm, '{0}_{1}'.format(self.ylabel, label), self.figs_path)


  def compare_plots(self, a, b, labels=['without', 'with']):
    """
    Plot roc curve, precision-recall curve and confussion matrices for the
    prediction of both models, i.e., without and with structural properties.

    :param a: Predicted probabilities for the model **without** the structural
      properties of the network.
    :type a: np.array[float]
    :param b: Predicted probabilities for the model **including** the
      structural properties of the network.
    :type b: np.array[float]
    :param labels: Labels of both models for the plots.
    :type labels: List[string]
    """
    # ROC AUC
    roc_a = roc_auc_score(self.y, a)
    fpr_a, tpr_a, _ = roc_curve(self.y, a)
    roc_b = roc_auc_score(self.y, b)
    fpr_b, tpr_b, _ = roc_curve(self.y, b)
    plot_rocs([fpr_a, fpr_b], [tpr_a, tpr_b], [roc_a, roc_b], labels, self.ylabel, self.figs_path)

    # Average precision
    ap_a = average_precision_score(self.y, a)
    prec_a, recall_a, thresh_a = precision_recall_curve(self.y, a)
    ap_b = average_precision_score(self.y, b)
    prec_b, recall_b, thresh_b = precision_recall_curve(self.y, b)
    plot_prs([recall_a,recall_b], [prec_a,prec_b], [ap_a,ap_b], labels, self.ylabel, self.figs_path)

    # compute best threshold, according to PR curve
    anew = self.opt_threshold(self.y, a)
    ancm = confusion_matrix(self.y, anew, normalize='true') # normalized confusion matrix
    plot_conf_matrix(ancm, '{0}_{1}'.format(self.ylabel, labels[0]), self.figs_path)
    bnew = self.opt_threshold(self.y, b)
    bncm = confusion_matrix(self.y, bnew, normalize='true') # normalized confusion matrix
    plot_conf_matrix(bncm, '{0}_{1}'.format(self.ylabel, labels[1]), self.figs_path)


  def write_csv(self, a, b, labels=['without', 'with']):
    """
    Save the evaluation metrics for prediction of both models, i.e., without
    and with structural properties.

    :param a: Predicted probabilities for the model **without** the structural
      properties of the network.
    :type a: np.array[float]
    :param b: Predicted probabilities for the model **including** the
      structural properties of the network.
    :type b: np.array[float]
    :param labels: Labels of both models for the plots.
    :type labels: List[string]
    """
    df_resume = pd.DataFrame()
    metrics = a.keys()
    df_resume['Metric'] = pd.Series(metrics)
    df_resume[labels[0]] = pd.Series([a[k] for k in metrics])
    df_resume[labels[1]] = pd.Series([b[k] for k in metrics])

    df_resume.to_csv('{0}/{1}_resume.csv'.format(self.output_path, self.ylabel), index=False)


  def train(self, X, n_splits=5, seed=None, n_iter=5, n_jobs_cv=None,
            n_jobs_xgb=2, eval_metric="aucpr", scoring="recall"):
    """
    Evaluate the performance of the prediction using metrics such as the
    auc roc, average precision score, precision, recall and F1 score.

    :param X: Iput dataset for the prediction.
    :type X: DataFrame
    :param n_splits: Number of folds for cross-validation, defaults to 5
    :type n_splits: int
    :param seed: Random number seed, defaults to None
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
    # Scores for each fold
    tpr, tnr = list(),list()
    fimp = dict([(c,0.0) for c in X.columns])
    params = dict()

    y_pred_prob = np.zeros(len(self.y))

    # Implementing SMOTE Technique, Cross Validating the right way
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in sss.split(X, self.y):
      Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
      ytrain, ytest = self.y.iloc[train_index], self.y.iloc[test_index]

      Xtrain, Xtest = Xtrain.values, Xtest.values # Turn into an array
      ytrain, ytest = ytrain.values, ytest.values

      # avoid folds without positive or negative class both in train and test
      if np.sum(ytest) == 0 or np.sum(ytest) == len(ytest): continue
      if np.sum(ytrain) == 0 or np.sum(ytrain) == len(ytrain): continue

      clf = self.create_classifier(n_iter=n_iter, n_jobs_cv=n_jobs_cv, n_jobs_xgb=n_jobs_xgb,
                          eval_metric=eval_metric, scoring=scoring, seed=seed)
      # pipeline for over-sampling and prediction. If the number of positive samples in
      # the training data is more than 3 smote is used, random oversampling otherwise
      try:
        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), clf)
        model = pipeline.fit(Xtrain, ytrain)
      except:
        pipeline = imbalanced_make_pipeline(RandomOverSampler(sampling_strategy='minority'), clf)
        model = pipeline.fit(Xtrain, ytrain)

      best_est = clf.best_estimator_
      pred_prob = best_est.predict_proba(Xtest)[:,1]
      y_pred_prob[test_index] = pred_prob # save results per fold

      pfm = self.evaluate(ytest, pred_prob) # performance of prediction for fold
      tpr.append(pfm['tpr']) # check -- not used
      tnr.append(pfm['tnr'])

      total_gain = np.sum(best_est.feature_importances_)
      feature_importances_ = best_est.feature_importances_ / total_gain
      for col, val in zip(X.columns, feature_importances_): # default importance type
        if not np.isnan(val):
          fimp[col] += val

      for p in clf.best_params_:
        if p not in params: params[p]=list()
        params[p]+=[clf.best_params_[p]]

    for c in X.columns: # check -- not necesary, the sum is enough
      fimp[c] /= n_splits

    for p in params:
      mode = stats.mode(params[p])
      params[p] = mode[0][0] if mode[1][0] > 1 else "Any"

    return y_pred_prob, fimp, params


  def structural_test(self, n_splits=5, seed=None, log=False, csv=True):
    """
    Test whether the structural properties of the network help to improve the
    prediction performance by building two different models and compare their
    results. One model includes the structural properties, whereas the other
    not.

    :param n_splits: Number of folds for cross-validation, defaults to 5
    :type n_splits: int
    :param seed: Random number seed, defaults to None
    :type seed: int
    :param log: Flag for logging of the results of the test, default to False
    :type log: bool
    :param csv: Flag for saving the results of the test in a csv file, defaults
      to True
    :type csv: bool

    :return: Result of the structural test (True if the the structural
      properties improve the prediction performance, False otherwise),
      predicted labels and best parameter combination for the classfifier
      including the structural properties.
    :rtype: Tuple(bool, np.array[Int], dict[string->float])
    """

    # prediction without structural properties
    df = self.df[self.orig_cols].copy()
    wo_pred_prob, wo_fimp, _ = self.train(df, n_splits=n_splits, seed=seed)
    wo_perf = self.evaluate(self.y, wo_pred_prob)
    if log: self.print_performance(wo_perf, 'Without structural properties')

    # prediction including structural properties
    strc_df = self.df[self.orig_cols+self.strc_cols].copy()
    strc_pred_prob, strc_fimp, strc_params = self.train(strc_df, n_splits=n_splits, seed=seed)
    strc_perf = self.evaluate(self.y, strc_pred_prob)
    strc_pred_clf = self.opt_threshold(self.y, strc_pred_prob)
    if log: self.print_performance(strc_pred_prob, 'Including topological properties')

    self.compare_plots(wo_pred_prob, strc_pred_prob)
    self.write_csv(wo_perf, strc_perf)

    # test whether the structural properties of the network improve the
    # performance of the prediction
    ans = (wo_perf['tpr'] + 0.05 < strc_perf['tpr']) and (wo_perf['tnr'] < strc_perf['tnr'])
    return ans, strc_pred_clf, strc_params
