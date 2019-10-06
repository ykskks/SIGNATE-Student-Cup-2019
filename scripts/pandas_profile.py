import os
import sys
import re
import logging

import pandas as pd
import pandas_profiling
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import category_encoders as ce
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

sys.path.append(".")
from utils import update_tracking, log_evaluation, preprocess_df


#################### 
## Changes
#################### 
MODEL_ID = "preprocess_df"


#################### 
## Load data
#################### 
# 変数名の英訳
train_cols_eng = ["id", "rent", "location", "access", "layout", "age", "direction", "area", "floor",
           "bath_toilet", "kitchen", "broad_com", "facility", "parking", "environment", "structure",
           "contract_period"]
test_cols_eng = ["id", "location", "access", "layout", "age", "direction", "area", "floor",
           "bath_toilet", "kitchen", "broad_com", "facility", "parking", "environment", "structure",
           "contract_period"]

train = pd.read_csv("./data/train.csv", names=train_cols_eng, header=0)
test = pd.read_csv("./data/test.csv", names=test_cols_eng, header=0)


#################### 
## Preprocess data
#################### 

train = preprocess_df(train)
print("Preprocessing done!")


#################### 
## Visualize
#################### 
profile = train.profile_report()
profile.to_file(f"./logs/visualization/{MODEL_ID}_profile.html")