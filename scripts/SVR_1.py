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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from utils import update_tracking, log_evaluation, preprocess_df


#################### 
## Changes
#################### 
# inheriting from TEINEI_4
MODEL_ID = "SVR_1"

# using SVR for an estimator



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
KERNEL = "rbf"
C = 1.0
EPS = 0.1


update_tracking(MODEL_ID, "kernel", KERNEL)
update_tracking(MODEL_ID, "C", C)
update_tracking(MODEL_ID, "epsilon", EPS)


params = {"kernel": KERNEL,
         "C": C,
         "epsilon": EPS}


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

train_processed = preprocess_df(train)
test_processed = preprocess_df(test)

# handle outliers
train_processed.drop(20427, axis=0, inplace=True) # 築1019年、どう修正するべきか不明なので
train_processed.loc[20231, "age_year"] = 52
train_processed.loc[20231, "age_in_months"] = 52 * 12 + 5 # 築520年、おそらく52年のタイポと仮定

train_processed.loc[5775, "rent"] = 120350 # 条件からしてありえない高値。おそらくゼロの個数違い
train_processed.loc[20926, "rent"] = 1160000 # 条件からしてありえない安値。おそらくゼロの個数違い

train_processed.reset_index(drop=True, inplace=True)
target = train_processed["rent"]
target_log = np.log1p(target)
train_processed.drop(["id", "rent"], axis=1, inplace=True)
test_processed.drop("id", axis=1, inplace=True)

#################### 
## get feature
#################### 
# モデル学習用データフレーム（category encoderの都合で分ける）
train_use = pd.DataFrame()
test_use = pd.DataFrame()

### location ###
ce_ordinal = ce.OneHotEncoder(cols=["district"], handle_missing="value")
train_use["district"] = train_processed["district"]
test_use["district"] = test_processed["district"]
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

### access ###
train_use["min_to_nearest_sta"] = train_processed["access_min"].apply(lambda x: min(x) if x else np.nan)
test_use["min_to_nearest_sta"] = test_processed["access_min"].apply(lambda x: min(x) if x else np.nan)

train_use["num_sta"] = train_processed["access_sta"].apply(lambda x: len(x))
test_use["num_sta"] = test_processed["access_sta"].apply(lambda x: len(x))


# 路線
line_cols = [col for col in train_processed.columns.values if "線" in col or "ライン" in col
                                                or "ライナー" in col or "エクスプレス" in col]

line_cols = [col for col in line_cols if train_processed[col].dropna().sum() > 300]
train_use[line_cols] = train_processed[line_cols]
test_use[line_cols] = test_processed[line_cols]

# 駅
sta_cols = [col for col in train_processed.columns.values if "駅" in col]

sta_cols = [col for col in sta_cols if train_processed[col].dropna().sum() > 300]
train_use[sta_cols] = train_processed[sta_cols]
test_use[sta_cols] = test_processed[sta_cols]


### layout ###
ce_ordinal = ce.OneHotEncoder(cols=["layout"], handle_missing="value")
train_use["layout"] = train_processed["layout"]
test_use["layout"] = test_processed["layout"]
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

layout_cols = ["is_K", "is_R", "is_L", "is_D", "is_S", "num_room"]

train_use[layout_cols] = train_processed[layout_cols]
test_use[layout_cols] = test_processed[layout_cols]



### age ###
age_cols = ["age_year", "age_month", "age_in_months"]
train_use[age_cols] = train_processed[age_cols]
test_use[age_cols] = test_processed[age_cols]

### direction ###
ce_ordinal = ce.OneHotEncoder(cols=["direction"], handle_missing="value")
train_use["direction"] = train_processed["direction"]
test_use["direction"] = test_processed["direction"]
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

direction_cols = ["has_N", "has_S", "has_E", "has_W"]
train_use[direction_cols] = train_processed[direction_cols]
test_use[direction_cols] = test_processed[direction_cols]


### area ###
train_use["area"] = train_processed["area"]
test_use["area"] = test_processed["area"]

train_use["area_per_room"] = train_use["area"] / train_use["num_room"]
test_use["area_per_room"] = test_use["area"] / test_use["num_room"]

### floor ###
train_processed["floor_ratio"] = train_processed["room_floor"] / train_processed["building_floor"]
test_processed["floor_ratio"] = test_processed["room_floor"] / test_processed["building_floor"]

floor_cols = ["has_underground", "room_floor", "building_floor", "floor_ratio"]
train_use[floor_cols] = train_processed[floor_cols]
test_use[floor_cols] = test_processed[floor_cols]

### bath_toilet ###
bath_toilet_cols = ["シャワー", "バスなし", "バス・トイレ別", "共同トイレ", "共同バス",
                    "専用トイレ", "専用バス", "洗面台独立", "浴室乾燥機", "温水洗浄便座", "脱衣所", "追焚機能"]

train_use[bath_toilet_cols] = train_processed[bath_toilet_cols]
test_use[bath_toilet_cols] = test_processed[bath_toilet_cols]


### kitchen ###
kitchen_cols = ["IHコンロ", "L字キッチン", "カウンターキッチン", "ガスコンロ", "コンロ1口", "コンロ2口", "コンロ3口",
                 "コンロ4口以上", "コンロ設置可（コンロ1口）", "コンロ設置可（コンロ2口）", "コンロ設置可（コンロ3口）",
                "コンロ設置可（コンロ4口以上）", "コンロ設置可（口数不明）", "システムキッチン", "冷蔵庫あり", "独立キッチン",
                  "給湯", "電気コンロ"]

train_use[kitchen_cols] = train_processed[kitchen_cols]
test_use[kitchen_cols] = test_processed[kitchen_cols]


### broad_com ###
broad_com_cols = ["BSアンテナ", "CATV", "CSアンテナ", "インターネット使用料無料",
                 "インターネット対応", "光ファイバー", "有線放送", "高速インターネット"]

train_use[broad_com_cols] = train_processed[broad_com_cols]
test_use[broad_com_cols] = test_processed[broad_com_cols]


### facility ###
facility_cols = ["24時間換気システム", "2面採光",
                 "3面採光", "ウォークインクローゼット", "エアコン付", "エレベーター", "オール電化", "ガスその他",
                "ガス暖房", "クッションフロア", "シューズボックス", "タイル張り", "トランクルーム", "バリアフリー",
                 "バルコニー", "フローリング", "プロパンガス", "ペアガラス", "ルーフバルコニー", "ロフト付き", "下水",
                "二世帯住宅", "二重サッシ", "井戸", "公営水道", "冷房", "出窓", "地下室", "室内洗濯機置場",
                 "室外洗濯機置場", "専用庭", "床下収納", "床暖房", "排水その他", "敷地内ごみ置き場", "水道その他",
                "汲み取り", "洗濯機置場なし", "浄化槽", "石油暖房", "都市ガス", "防音室"]

train_use[facility_cols] = train_processed[facility_cols]
test_use[facility_cols] = test_processed[facility_cols]


### parking ### 
parking_cols = ["bicycle_parking", "car_parking", "bike_parking"]
train_use[parking_cols] = train_processed[parking_cols]
test_use[parking_cols] = test_processed[parking_cols]



### environment ###
env_cols = ["デパート", "公園",
             "郵便局", "コインパーキング", "学校", "図書館", "飲食店", "月極駐車場", "銀行", "小学校",
             "ドラッグストア", "レンタルビデオ", "病院", "総合病院", "コンビニ", "大学", "幼稚園・保育園",
            "スーパー", "クリーニング"]

train_use[env_cols] = train_processed[env_cols]
test_use[env_cols] = test_processed[env_cols]


### structure ###
ce_ordinal = ce.OneHotEncoder(cols=["structure"], handle_missing="value")
train_use["structure"] = train_processed["structure"]
test_use["structure"] = test_processed["structure"]
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

### contract_period ###
period_cols = ["fixed_term", "contract_period_year", "contract_period_month", "contract_period_in_months"]
train_use[period_cols] = train_processed[period_cols]
test_use[period_cols] = test_processed[period_cols]


# nan handling
use_cols = train_use.columns.values
for col in use_cols:
	train_use[col].fillna(-1, inplace=True)
	test_use[col].fillna(-1, inplace=True)


# scaling 
sc = StandardScaler()
train_use = sc.fit_transform(train_use)
test_use = sc.transform(test_use)

logger.debug(f"Using features:{use_cols}")




#################### 
## Train model
#################### 
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_use))
predictions = np.zeros(len(test_use))


for fold, (train_idx, val_idx) in enumerate(folds.split(train_use)):
    print(f"Fold {fold+1}")
    x_train, y_train = train_use[train_idx], target_log[train_idx]
    x_val, y_val = train_use[val_idx], target_log[val_idx]
    
    model = SVR(**params)
    model.fit(x_train, y_train)

    oof[val_idx] = model.predict(x_val)

    logger.debug("##### train rmse #####")
    predicted_train = model.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(np.expm1(predicted_train), np.expm1(y_train)))
    logger.debug(train_rmse)
    
    predictions += model.predict(test_use) / folds.n_splits
    
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