import os
import sys
import re
import logging

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import category_encoders as ce
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

sys.path.append(".")
from utils import update_tracking, log_evaluation


#################### 
## Changes
#################### 
MODEL_ID = "BASE_10"
# LB worse than CV > stronger regularization(lambda l1, lamdba l2)
# max_depth, num_leaves=5
# col_sample_by_tree=0.7

# more feature
# num_sta: number of stations nearby
# has_nando, is_R, is_K, is_DK, is_LDK
# age_month, age(年と月合わせて月表記にしたもの))
# floor_room_over_floor_building: 建物の中での相対的な位置
# それぞれの有り無しではなく、その数を数える(num_bath, num_facility etc.)

# longer training (n_rounds = 30000)




logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
sc = logging.StreamHandler()
logger.addHandler(sc)
fh = logging.FileHandler(f"./logs/model_logs/{MODEL_ID}.log")
logger.addHandler(fh)
logger.debug(f"./logs/model_logs/{MODEL_ID}.log")


#################### 
## Parameters
#################### 
N_ROUNDS = 30000
LR = 0.01
BOOSTING = "gbdt"
BAG_FREQ = 1
BAG_FRAC = 0.7
MIN_DATA_IN_LEAF = 50
SEED = 42
METRIC = "rmse"
L1 = 1e-2
L2 = 1e-2
MAX_DEPTH = 5
FEAT_FRAC = 0.7

update_tracking(MODEL_ID, "n_rounds", N_ROUNDS)
update_tracking(MODEL_ID, "lr", LR)
update_tracking(MODEL_ID, "boosting", BOOSTING)
update_tracking(MODEL_ID, "bag_freq", BAG_FREQ)
update_tracking(MODEL_ID, "bag_frac", BAG_FRAC)
update_tracking(MODEL_ID, "min_data_in_leaf", MIN_DATA_IN_LEAF)
update_tracking(MODEL_ID, "seed", SEED)
update_tracking(MODEL_ID, "metric", METRIC)
update_tracking(MODEL_ID, "lambda_l1", L1)
update_tracking(MODEL_ID, "lambda_l2", L2)
update_tracking(MODEL_ID, "max_depth", MAX_DEPTH)
update_tracking(MODEL_ID, "feature_fraction", FEAT_FRAC)


params = {"learning_rate": LR,
         "boosting": BOOSTING,
         "bagging_freq": BAG_FREQ,
         "bagging_fraction": BAG_FRAC,
         "min_data_in_leaf": MIN_DATA_IN_LEAF,
         "bagging_seed": SEED,
         "metric": METRIC,
         "random_state": SEED,
         "lambda_l1": L1,
         "lambda_l2": L2,
         "max_depth": MAX_DEPTH,
         "feature_fraction": FEAT_FRAC}


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

train_id = train["id"]
test_id = test["id"]
target = train["rent"] 
log_target = np.log1p(target) # log transformation
train.drop(["id", "rent"], axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

use_cols = []

#################### 
## Preprocess data
#################### 

### location ###
train["districts"] = train["location"].apply(lambda x: re.search("(?<=都)(.*?)(?=区)", x).group())
test["districts"] = test["location"].apply(lambda x: re.search("(?<=都)(.*?)(?=区)", x).group())

ce_ordinal = ce.OrdinalEncoder(cols=["districts"], handle_missing="value")
train = ce_ordinal.fit_transform(train)
test = ce_ordinal.transform(test)
use_cols.append("districts")

### access ###
train["mins_to_nearest_sta"] = train["access"].apply(lambda x: min(map(int, re.findall("(?<=徒歩)(.*?)(?=分)", x))))
test["mins_to_nearest_sta"] = test["access"].apply(lambda x: min(map(int, re.findall("(?<=徒歩)(.*?)(?=分)", x))))
train["num_sta"] = train["access"].apply(lambda x: len(re.findall("(?<=徒歩)(.*?)(?=分)", x))) 
test["num_sta"] = test["access"].apply(lambda x: len(re.findall("(?<=徒歩)(.*?)(?=分)", x))) 
use_cols.extend(["mins_to_nearest_sta", "num_sta"])

### layout ###
train["num_room"] = train["layout"].apply(lambda x: int(re.search("[0-9]", x).group()))
test["num_room"] = test["layout"].apply(lambda x: int(re.search("[0-9]", x).group()))
train["has_nando"] = train["layout"].apply(lambda x: 1 if "納戸" in x else 0)
test["has_nando"] = test["layout"].apply(lambda x: 1 if "納戸" in x else 0)
train["is_R"] = train["layout"].apply(lambda x: 1 if "R" in x else 0)
test["is_R"] = test["layout"].apply(lambda x: 1 if "R" in x else 0)
train["is_K"] = train["layout"].apply(lambda x: 1 if "K" in x else 0)
test["is_K"] = test["layout"].apply(lambda x: 1 if "K" in x else 0)
train["is_DK"] = train["layout"].apply(lambda x: 1 if "DK" in x else 0)
test["is_DK"] = test["layout"].apply(lambda x: 1 if "DK" in x else 0)
train["is_LDK"] = train["layout"].apply(lambda x: 1 if "LDK" in x else 0)
test["is_LDK"] = test["layout"].apply(lambda x: 1 if "LDK" in x else 0)
use_cols.extend(["num_room", "has_nando", "is_R", "is_K", "is_DK", "is_LDK"])

### age ###
train["age_year"] = train["age"].apply(lambda x: int(re.match("[0-9]*(?=年)", x).group()) if x != "新築" else 0)
test["age_year"] = test["age"].apply(lambda x: int(re.match("[0-9]*(?=年)", x).group()) if x != "新築" else 0)
train["age_month"] = train["age"].apply(lambda x: int(re.search("[0-9]*(?=ヶ月)", x).group()) if x != "新築" else 0)
test["age_month"] = test["age"].apply(lambda x: int(re.search("[0-9]*(?=ヶ月)", x).group()) if x != "新築" else 0)
train["age"] = train["age_year"] * 12 + train["age_month"]
test["age"] = test["age_year"] * 12 + test["age_month"]
use_cols.extend(["age_year", "age_month", "age"])

### direction ###
ce_ordinal = ce.OrdinalEncoder(cols=["direction"], handle_missing="value")
train = ce_ordinal.fit_transform(train)
test = ce_ordinal.transform(test)
use_cols.append("direction")

### area ###
train["area"] = train["area"].apply(lambda x: float(x[:-2]))
test["area"] = test["area"].apply(lambda x: float(x[:-2]))
use_cols.append("area")

### floor ###
# testに一つだけ欠損有り, trainのmedianで埋める
train["floor_is_underground"] = train["floor"].apply(lambda x: int(bool(re.match("^地下.*", x))))
test["floor_is_underground"] = test["floor"].apply(lambda x: int(bool(re.match("^地下.*", x))) if not pd.isnull(x) else 0)

train["floor_room"] = train["floor"].apply(lambda x: re.match("^[0-9][0-9]*", x).group() if re.match("^[0-9][0-9]*", x) else 0)
test["floor"] = test["floor"].apply(lambda x: x if not pd.isnull(x) else str(train["floor_room"].median()))
test["floor_room"] = test["floor"].apply(lambda x: re.match("^[0-9][0-9]*", x).group() if re.match("^[0-9][0-9]*", x) else 0)
train["floor_room"] = train["floor_room"].astype(int)
test["floor_room"] = test["floor_room"].astype(int)

def clean_floor_list(lis):
    if len(lis) == 1:
        return int(lis[0])
    elif len(lis) >= 2:
        return int(lis[1])

train["floor_list"] = train["floor"].apply(lambda x: list(filter(lambda x: x != "", re.findall("[0-9]*", x))))
train["floor_building"] = train["floor_list"].apply(clean_floor_list)
test["floor_list"] = test["floor"].apply(lambda x: list(filter(lambda x: x != "", re.findall("[0-9]*", x))) if not pd.isnull(x) else train["floor_building"].median())
test["floor_building"] = test["floor_list"].apply(clean_floor_list)

train["floor_room_over_floor_building"] = train["floor_room"] / train["floor_building"]
test["floor_room_over_floor_building"] = test["floor_room"] / test["floor_building"]

use_cols.extend(["floor_is_underground", "floor_room", "floor_building", "floor_room_over_floor_building"])

### bath_toilet ###
bath_toilet_train = train["bath_toilet"].str.get_dummies(sep="／\t")
bath_toilet_test = test["bath_toilet"].str.get_dummies(sep="／\t")

train["num_bath_toilet"] = train["bath_toilet"].str.count("／\t") + 1
test["num_bath_toilet"] = test["bath_toilet"].str.count("／\t") + 1

train = pd.concat([train, bath_toilet_train], axis=1)
test = pd.concat([test, bath_toilet_test], axis=1)
test.drop("トイレなし", axis=1, inplace=True) # テストにしかない

use_cols.append("num_bath_toilet")
use_cols.extend(list(bath_toilet_train.columns.values))


### kitchen ###
# "\t／\t", "／\t"が混じってる
train["kitchen"] = train["kitchen"].str.replace("\t／\t", "／\t")
test["kitchen"] = test["kitchen"].str.replace("\t／\t", "／\t")

kitchen_train = train["kitchen"].str.get_dummies(sep="／\t")
kitchen_test = test["kitchen"].str.get_dummies(sep="／\t")

train["num_kitchen"] = train["kitchen"].str.count("／\t") + 1
test["num_kitchen"] = test["kitchen"].str.count("／\t") + 1

train = pd.concat([train, kitchen_train], axis=1)
test = pd.concat([test, kitchen_test], axis=1)

use_cols.append("num_kitchen")
use_cols.extend(list(kitchen_train.columns.values))

### broad_com ###
broad_com_train = train["broad_com"].str.get_dummies(sep="／\t")
broad_com_test = test["broad_com"].str.get_dummies(sep="／\t")

train["num_broad_com"] = train["broad_com"].str.count("／\t") + 1
test["num_broad_com"] = test["broad_com"].str.count("／\t") + 1

train = pd.concat([train, broad_com_train], axis=1)
test = pd.concat([test, broad_com_test], axis=1)

use_cols.append("num_broad_com")
use_cols.extend(list(broad_com_train.columns.values))


### facility ###
# "／\t", "\t"が混じってる
train["facility"] = train["facility"].str.replace("／\t", "\t")
test["facility"] = test["facility"].str.replace("／\t", "\t")

facility_train = train["facility"].str.get_dummies(sep="\t")
facility_test = test["facility"].str.get_dummies(sep="\t")

train["num_facility"] = train["facility"].str.count("\t") + 1
test["num_facility"] = test["facility"].str.count("\t") + 1

train = pd.concat([train, facility_train], axis=1)
test = pd.concat([test, facility_test], axis=1)

use_cols.append("num_facility")
use_cols.extend(list(facility_train.columns.values))


### parking ### 
def bicycle_parking(parking):
    if pd.isnull(parking):
        return np.nan
    elif re.search("駐輪場.*?有", parking):
        return 1
    else:
        return 0

def car_parking(parking):
    if pd.isnull(parking):
        return np.nan
    elif re.search("駐車場.*?有", parking):
        return 1
    else:
        return 0

def bike_parking(parking):
    if pd.isnull(parking):
        return np.nan
    elif re.search("バイク置き場.*?有", parking):
        return 1
    else:
        return 0
    
train["bicycle_parking"] = train["parking"].apply(bicycle_parking)
test["bicycle_parking"] = test["parking"].apply(bicycle_parking)
use_cols.append("bicycle_parking")

train["car_parking"] = train["parking"].apply(car_parking)
test["car_parking"] = test["parking"].apply(car_parking)
use_cols.append("car_parking")

train["bike_parking"] = train["parking"].apply(bike_parking)
test["bike_parking"] = test["parking"].apply(bike_parking)
use_cols.append("bike_parking")

### environment ###
def clean_environment(environment):
    if pd.isnull(environment):
        return np.nan
    else:
    
        return re.findall("(?<=【).*?(?=】)", environment)

train["environment_list"] = train["environment"].apply(clean_environment)
test["environment_list"] = test["environment"].apply(clean_environment)

train["num_environment"] = train["environment"].str.count("\t") + 1
test["num_environment"] = test["environment"].str.count("\t") + 1

env_uniq = []
for lis in train["environment_list"].dropna():
    for i in lis:
        env_uniq.append(i)
        
env_uniq = list(set(env_uniq))

for col in env_uniq:
    train[col] = 0
    test[col] = 0
    train[col] = train["environment"].apply(lambda x: int(re.search(f"(?<=【{col}】 ).*?(?=m)", x).group()) if (not pd.isnull(x) and re.search(f"(?<=【{col}】 ).*?(?=m)", x)) else np.nan)
    test[col] = test["environment"].apply(lambda x: int(re.search(f"(?<=【{col}】 ).*?(?=m)", x).group()) if (not pd.isnull(x) and re.search(f"(?<=【{col}】 ).*?(?=m)", x)) else np.nan)

use_cols.append("num_environment")
use_cols.extend(env_uniq)

### structure ###
ce_ordinal = ce.OrdinalEncoder(cols=["structure"], handle_missing="value")
train = ce_ordinal.fit_transform(train)
test = ce_ordinal.transform(test)
use_cols.append("structure")

### contract_period ###
def is_fixed_term(period):
    if pd.isnull(period):
        return np.nan
    elif "定期借家" in period:
        return 1
    else:
        return 0 

train["fixed_term"] = train["contract_period"].apply(is_fixed_term)
test["fixed_term"] = test["contract_period"].apply(is_fixed_term)
use_cols.append("fixed_term")

def check_contract_period(period):
    if pd.isnull(period):
        return np.nan
    elif re.match("[0-9]+(?=年間)", period):
        return int(re.match(".*?(?=年間)", period).group())
    elif re.match("[0-9]+(?=ヶ月間)", period):
        return float(re.match(".*?(?=ヶ月間)", period).group()) # 4年8ヶ月とかがあるが。。
    else:
        return np.nan # 2021年4月まで	※この物件は	定期借家	です。

train["contract_period"] = train["contract_period"].apply(check_contract_period)
test["contract_period"] = test["contract_period"].apply(check_contract_period)
use_cols.append("contract_period")

logger.debug(f"Using features:{use_cols}")

categorical_cols = ["districts", "has_nando", "is_R", "is_K", "is_DK", "is_LDK",
                    "direction", "floor_is_underground", "シャワー", "バスなし", "バス・トイレ別", 
                    "共同トイレ", "共同バス", "専用トイレ", "専用バス", "洗面台独立", "浴室乾燥機", "浴室乾燥機\t", 
                    "温水洗浄便座", "脱衣所", "追焚機能", "IHコンロ", "L字キッチン", "カウンターキッチン", "ガスコンロ", 
                    "コンロ1口", "コンロ2口", "コンロ3口", "コンロ4口以上", "コンロ設置可（コンロ1口）", 
                    "コンロ設置可（コンロ2口）", "コンロ設置可（コンロ3口）", "コンロ設置可（コンロ4口以上）",
                    "コンロ設置可（口数不明）", "システムキッチン", "冷蔵庫あり", "独立キッチン", "給湯", "電気コンロ", 
                    "BSアンテナ", "CATV", "CSアンテナ", "インターネット使用料無料", "インターネット対応", "光ファイバー", 
                    "有線放送", "高速インターネット", "24時間換気システム", "2面採光", "3面採光", 
                    "ウォークインクローゼット", "エアコン付", "エレベーター", "オール電化", "ガスその他", "ガス暖房", 
                    "クッションフロア", "シューズボックス", "タイル張り", "トランクルーム", "バリアフリー", "バルコニー",
                    "フローリング", "プロパンガス", "ペアガラス", "ルーフバルコニー", "ロフト付き", "下水", "二世帯住宅", 
                    "二重サッシ", "井戸", "公営水道", "冷房", "出窓", "地下室", "室内洗濯機置場", "室外洗濯機置場", 
                    "専用庭", "床下収納", "床暖房", "排水その他", "敷地内ごみ置き場", "水道その他", "汲み取り", 
                    "洗濯機置場なし", "浄化槽", "石油暖房", "都市ガス", "防音室", "bicycle_parking", "car_parking", 
                    "bike_parking", "structure", "fixed_term"]

#################### 
## Train model
#################### 
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(folds.split(train)):
    print(f"Fold {fold+1}")
    train_data = lgb.Dataset(train.iloc[train_idx][use_cols], label=log_target[train_idx], categorical_feature=categorical_cols)
    val_data = lgb.Dataset(train.iloc[val_idx][use_cols], label=log_target[val_idx], categorical_feature=categorical_cols)
    num_round = N_ROUNDS
    callbacks = [log_evaluation(logger, period=100)]
    clf = lgb.train(params, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=False, early_stopping_rounds=100, callbacks=callbacks)
    oof[val_idx] = clf.predict(train[use_cols].values[val_idx], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = use_cols
    fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    feature_importance_df = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).head(50)
    logger.debug("##### feature importance #####")
    logger.debug(feature_importance_df)
    
    predictions += clf.predict(test[use_cols], num_iteration=clf.best_iteration) / folds.n_splits
    
# inverse log transformation
oof = np.expm1(oof)
predictions = np.expm1(predictions)


cv_score = np.sqrt(mean_squared_error(oof, target))
logger.debug(f"5fold CV score: {cv_score}")
update_tracking(MODEL_ID, "cv_rmse", cv_score)



#################### 
## Submit
#################### 
spsbm = pd.read_csv("./data/sample_submit.csv", header=None)
spsbm.iloc[:, 1] = predictions
spsbm.to_csv(f"./submissions/{MODEL_ID}.csv", header=None, index=None)