#!/usr/bin/python3
# coding: utf-8

# Hierarchical node classification (attribute prediction considering hierarchy of classes)
# Test of algorithm
# Miguel Romero, 2021 ago 12

import os
import sys
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

# XGBhc library
from context import xgbhc
from xgbhc import XGBhc


def readListFile(filename):
  """
  Save the evaluation metrics for prediction of both models, i.e., without
  and with structural properties.

  :param filename: Predicted probabilities for the model.
  :type filename: string

  :return: List of strings
  :rtype: list[string]
  """
  file = open(filename, 'r')
  tmp_list = [x.strip() for x in file.readlines()]
  file.close()
  return tmp_list


###############################################
# Gene function prediction (biological process)
###############################################

start_time = time()
dt = datetime.datetime.today()

PATH = "/home/miguel/projects/omics/XGBhc/test"
DATA_PATH = "{0}/{1}".format(PATH, "data")
OUTPUT_PATH = "{0}/{1}".format(PATH, "output")

root = "GO:0002376"

hierarchy = readListFile("{0}/hierarchy.txt".format(DATA_PATH))
ancestors = readListFile("{0}/ancestors.txt".format(DATA_PATH))
data = pd.read_csv('{0}/data.csv'.format(DATA_PATH), dtype='float')

#########################
# Hierarchical prediction
#########################

model = XGBhc()
model.load_data(data, root, hierarchy, ancestors, DATA_PATH, OUTPUT_PATH)
model.train_hierarchy()

print("--- {0:.2f} seconds ---".format(time() - start_time))
