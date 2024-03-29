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
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

sys.path.append(".")
from utils import update_tracking, log_evaluation, preprocess_df


#################### 
## Changes
#################### 
MODEL_ID = "HAGIBIS_1"

##### 重複の処理 #####
# (trainのみに同じ物件 -> groupKFoldするのでそのまま)
# testのみに同じ物件 -> 予測値を平均取る
# trainとtestに同じ物件あり -> post_processでtrainに合わせる。

# 同じ建物で違う部屋のやつはまとめてGroupKfoldするべきな気がするがしてない



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
FEAT_FRAC = 0.9

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

use_cols = []

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
train_processed.loc[20926, "area"] = 43.01 # 条件からしてありえなく広い。おそらくゼロの個数違い


train_processed["ku"] = train_processed["location"].apply(lambda x: re.search("(?<=都).*?区", x).group())
train_processed["group"] = train_processed["ku"] + train_processed["building_floor"].astype(str) \
                    + train_processed["age_in_months"].astype(str) + train_processed["area"].astype(str)

# testの予測値置きかえ用
rent_dic = train_processed.groupby("group")["rent"].mean()


test_processed["ku"] = test_processed["location"].apply(lambda x: re.search("(?<=都).*?区", x).group())
test_group = test_processed["ku"] + test_processed["building_floor"].astype(str) \
                    + test_processed["age_in_months"].astype(str) + test_processed["area"].astype(str)

train_processed.reset_index(drop=True, inplace=True)
target = train_processed["rent"]
train_processed.drop(["id", "rent"], axis=1, inplace=True)
test_processed.drop("id", axis=1, inplace=True)


#################### 
## get feature
#################### 
# モデル学習用データフレーム（category encoderの都合で分ける）
train_use = pd.DataFrame()
test_use = pd.DataFrame()

### location ###
ce_ordinal = ce.OrdinalEncoder(cols=["district"], handle_missing="value")
train_use["district"] = train_processed["district"]
test_use["district"] = test_processed["district"]
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

# 緯度経度
geo_csvs = os.listdir("./data/geo/")
geo_csvs = [csv for csv in geo_csvs if "csv" in csv]
loc_dic = {}

for csv in geo_csvs:
    df = pd.read_csv("./data/geo/"+csv, encoding="shift-jis")
    df["loc"] = df["緯度"].astype(str) + "," + df["経度"].astype(str)
    dic = dict(zip(df["大字町丁目名"].values, df["loc"].values))
    loc_dic[df["市区町村名"].unique()[0]] = dic

train_processed["ku"] = train_processed["location"].apply(lambda x: re.search("(?<=都).*?区", x).group())
test_processed["ku"] = test_processed["location"].apply(lambda x: re.search("(?<=都).*?区", x).group())
train_processed["tyou"] = train_processed["location"].apply(lambda x: re.search("(?<=区).*?丁目", x).group() \
                                                            if re.search("(?<=区).*?丁目", x) else np.nan)
test_processed["tyou"] = test_processed["location"].apply(lambda x: re.search("(?<=区).*?丁目", x).group() \
                                                            if re.search("(?<=区).*?丁目", x) else np.nan)

num_map = {"１":"一", "２":"二", "３":"三", "４":"四", "５":"五", "６":"六", "７":"七", "８":"八", "９":"九"}

def convert_number(tyou):
    if pd.isnull(tyou):
        return np.nan
    
    for num in num_map.keys():
        if num in tyou:
            return tyou.replace(num, num_map[num])
        
train_processed["tyou"] = train_processed["tyou"].apply(convert_number)
test_processed["tyou"] = test_processed["tyou"].apply(convert_number)
train_processed["loc_processed"] = train_processed["ku"] + "," + train_processed["tyou"]
test_processed["loc_processed"] = test_processed["ku"] + "," + test_processed["tyou"]

def get_long_lati(loc_processed):
    if pd.isnull(loc_processed):
        return np.nan
    ku, chou = loc_processed.split(",")
    if chou in loc_dic[ku]:
        return loc_dic[ku][chou]
    else:
        return np.nan
    
# 丁目の情報がないのがほとんどnanの原因でいくつかはとってきたcsvにその丁目の情報なし
train_processed["lati_long"] = train_processed["loc_processed"].apply(get_long_lati)
test_processed["lati_long"] = test_processed["loc_processed"].apply(get_long_lati)
train_use["lati"] = train_processed["lati_long"].apply(lambda x: float(x.split(",")[0]) if not pd.isnull(x) else np.nan)
train_use["long"] = train_processed["lati_long"].apply(lambda x: float(x.split(",")[1]) if not pd.isnull(x) else np.nan)
test_use["lati"] = test_processed["lati_long"].apply(lambda x: float(x.split(",")[0]) if not pd.isnull(x) else np.nan)
test_use["long"] = test_processed["lati_long"].apply(lambda x: float(x.split(",")[1]) if not pd.isnull(x) else np.nan)

### access ###
train_use["min_to_nearest_sta"] = train_processed["access_min"].apply(lambda x: min(x) if x else np.nan)
test_use["min_to_nearest_sta"] = test_processed["access_min"].apply(lambda x: min(x) if x else np.nan)

train_use["num_sta"] = train_processed["access_sta"].apply(lambda x: len(x))
test_use["num_sta"] = test_processed["access_sta"].apply(lambda x: len(x))

train_nearest_idx = train_processed["access_min"].apply(lambda lis: np.argmin(np.array(lis)) if lis else np.nan)
train_nearest_sta = []
train_sta_list = train_processed["access_sta"].values
for i in range(len(train_processed)):
    if np.isnan(train_nearest_idx[i]):
        train_nearest_sta.append(np.nan)
    else:
        train_nearest_sta.append(train_sta_list[i][int(train_nearest_idx[i])])
train_use["nearest_sta"] = train_nearest_sta

test_nearest_idx = test_processed["access_min"].apply(lambda lis: np.argmin(np.array(lis)) if lis else np.nan)
test_nearest_sta = []
test_sta_list = test_processed["access_sta"].values
for i in range(len(test_processed)):
    if np.isnan(test_nearest_idx[i]):
        test_nearest_sta.append(np.nan)
    else:
        test_nearest_sta.append(test_sta_list[i][int(test_nearest_idx[i])])
test_use["nearest_sta"] = test_nearest_sta

ce_ordinal = ce.OrdinalEncoder(cols=["nearest_sta"], handle_missing="value")
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)


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
ce_ordinal = ce.OrdinalEncoder(cols=["layout"], handle_missing="value")
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
ce_ordinal = ce.OrdinalEncoder(cols=["direction"], handle_missing="value")
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
ce_ordinal = ce.OrdinalEncoder(cols=["structure"], handle_missing="value")
train_use["structure"] = train_processed["structure"]
test_use["structure"] = test_processed["structure"]
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

### contract_period ###
period_cols = ["fixed_term", "contract_period_year", "contract_period_month", "contract_period_in_months"]
train_use[period_cols] = train_processed[period_cols]
test_use[period_cols] = test_processed[period_cols]



# 組み合わせ特徴
train_use["area_over_age_in_months"] = train_use["area"] / (train_use["age_in_months"] + 1)
test_use["area_over_age_in_months"] = test_use["area"] / (test_use["age_in_months"] + 1)

train_use["district_layout"] = train_processed["district"].astype(str) + "_" + train_processed["layout"].astype(str)
test_use["district_layout"] = test_processed["district"].astype(str) + "_" + test_processed["layout"].astype(str)
ce_ordinal = ce.OrdinalEncoder(cols=["district_layout"], handle_missing="value")
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

train_use["district_structure"] = train_processed["district"].astype(str) + "_" + train_processed["structure"].astype(str)
test_use["district_structure"] = test_processed["district"].astype(str) + "_" + test_processed["structure"].astype(str)
ce_ordinal = ce.OrdinalEncoder(cols=["district_structure"], handle_missing="value")
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

train_use["layout_structure"] = train_processed["layout"].astype(str) + "_" + train_processed["structure"].astype(str)
test_use["layout_structure"] = test_processed["layout"].astype(str) + "_" + test_processed["structure"].astype(str)
ce_ordinal = ce.OrdinalEncoder(cols=["layout_structure"], handle_missing="value")
train_use = ce_ordinal.fit_transform(train_use)
test_use = ce_ordinal.transform(test_use)

train_use["area_room_floor"] = train_use["area"] * train_use["room_floor"]
test_use["area_room_floor"] = test_use["area"] * test_use["room_floor"]

train_use["area_building_floor"] = train_use["area"] * train_use["building_floor"]
test_use["area_building_floor"] = test_use["area"] * test_use["building_floor"]

train_use["lati_long"] = train_use["lati"] * train_use["long"]
train_use["lati_over_long"] = train_use["lati"] / train_use["long"]
train_use["long_over_lati"] = train_use["long"] / train_use["lati"]
test_use["lati_long"] = test_use["lati"] * test_use["long"]
test_use["lati_over_long"] = test_use["lati"] / test_use["long"]
test_use["long_over_lati"] = test_use["long"] / test_use["lati"]

# aggregation系特徴
train_use["area_mean_by_sta"] = train_use.groupby("nearest_sta")["area"].transform("mean")
train_use["area_max_by_sta"] = train_use.groupby("nearest_sta")["area"].transform("max")
train_use["area_min_by_sta"] = train_use.groupby("nearest_sta")["area"].transform("min")
train_use["area_std_by_sta"] = train_use.groupby("nearest_sta")["area"].transform("std")

train_use["area_mean_by_district"] = train_use.groupby("district")["area"].transform("mean")
train_use["area_max_by_district"] = train_use.groupby("district")["area"].transform("max")
train_use["area_min_by_district"] = train_use.groupby("district")["area"].transform("min")
train_use["area_std_by_district"] = train_use.groupby("district")["area"].transform("std")

train_use["area_mean_by_layout"] = train_use.groupby("layout")["area"].transform("mean")
train_use["area_max_by_layout"] = train_use.groupby("layout")["area"].transform("max")
train_use["area_min_by_layout"] = train_use.groupby("layout")["area"].transform("min")
train_use["area_std_by_layout"] = train_use.groupby("layout")["area"].transform("std")

train_use["area_mean_by_structure"] = train_use.groupby("structure")["area"].transform("mean")
train_use["area_max_by_structure"] = train_use.groupby("structure")["area"].transform("max")
train_use["area_min_by_structure"] = train_use.groupby("structure")["area"].transform("min")
train_use["area_std_by_structure"] = train_use.groupby("structure")["area"].transform("std")

train_use["age_mean_by_sta"] = train_use.groupby("nearest_sta")["age_in_months"].transform("mean")
train_use["age_max_by_sta"] = train_use.groupby("nearest_sta")["age_in_months"].transform("max")
train_use["age_min_by_sta"] = train_use.groupby("nearest_sta")["age_in_months"].transform("min")
train_use["age_std_by_sta"] = train_use.groupby("nearest_sta")["age_in_months"].transform("std")

train_use["age_mean_by_district"] = train_use.groupby("district")["age_in_months"].transform("mean")
train_use["age_max_by_district"] = train_use.groupby("district")["age_in_months"].transform("max")
train_use["age_min_by_district"] = train_use.groupby("district")["age_in_months"].transform("min")
train_use["age_std_by_district"] = train_use.groupby("district")["age_in_months"].transform("std")

train_use["age_mean_by_layout"] = train_use.groupby("layout")["age_in_months"].transform("mean")
train_use["age_max_by_layout"] = train_use.groupby("layout")["age_in_months"].transform("max")
train_use["age_min_by_layout"] = train_use.groupby("layout")["age_in_months"].transform("min")
train_use["age_std_by_layout"] = train_use.groupby("layout")["age_in_months"].transform("std")

train_use["age_mean_by_structure"] = train_use.groupby("structure")["age_in_months"].transform("mean")
train_use["age_max_by_structure"] = train_use.groupby("structure")["age_in_months"].transform("max")
train_use["age_min_by_structure"] = train_use.groupby("structure")["age_in_months"].transform("min")
train_use["age_std_by_structure"] = train_use.groupby("structure")["age_in_months"].transform("std")

train_use["building_floor_mean_by_sta"] = train_use.groupby("nearest_sta")["building_floor"].transform("mean")
train_use["building_floor_max_by_sta"] = train_use.groupby("nearest_sta")["building_floor"].transform("max")
train_use["building_floor_min_by_sta"] = train_use.groupby("nearest_sta")["building_floor"].transform("min")
train_use["building_floor_std_by_sta"] = train_use.groupby("nearest_sta")["building_floor"].transform("std")

train_use["building_floor_mean_by_district"] = train_use.groupby("district")["building_floor"].transform("mean")
train_use["building_floor_max_by_district"] = train_use.groupby("district")["building_floor"].transform("max")
train_use["building_floor_min_by_district"] = train_use.groupby("district")["building_floor"].transform("min")
train_use["building_floor_std_by_district"] = train_use.groupby("district")["building_floor"].transform("std")


test_use["area_mean_by_sta"] = test_use.groupby("nearest_sta")["area"].transform("mean")
test_use["area_max_by_sta"] = test_use.groupby("nearest_sta")["area"].transform("max")
test_use["area_min_by_sta"] = test_use.groupby("nearest_sta")["area"].transform("min")
test_use["area_std_by_sta"] = test_use.groupby("nearest_sta")["area"].transform("std")

test_use["area_mean_by_district"] = test_use.groupby("district")["area"].transform("mean")
test_use["area_max_by_district"] = test_use.groupby("district")["area"].transform("max")
test_use["area_min_by_district"] = test_use.groupby("district")["area"].transform("min")
test_use["area_std_by_district"] = test_use.groupby("district")["area"].transform("std")

test_use["area_mean_by_layout"] = test_use.groupby("layout")["area"].transform("mean")
test_use["area_max_by_layout"] = test_use.groupby("layout")["area"].transform("max")
test_use["area_min_by_layout"] = test_use.groupby("layout")["area"].transform("min")
test_use["area_std_by_layout"] = test_use.groupby("layout")["area"].transform("std")

test_use["area_mean_by_structure"] = test_use.groupby("structure")["area"].transform("mean")
test_use["area_max_by_structure"] = test_use.groupby("structure")["area"].transform("max")
test_use["area_min_by_structure"] = test_use.groupby("structure")["area"].transform("min")
test_use["area_std_by_structure"] = test_use.groupby("structure")["area"].transform("std")

test_use["age_mean_by_sta"] = test_use.groupby("nearest_sta")["age_in_months"].transform("mean")
test_use["age_max_by_sta"] = test_use.groupby("nearest_sta")["age_in_months"].transform("max")
test_use["age_min_by_sta"] = test_use.groupby("nearest_sta")["age_in_months"].transform("min")
test_use["age_std_by_sta"] = test_use.groupby("nearest_sta")["age_in_months"].transform("std")

test_use["age_mean_by_district"] = test_use.groupby("district")["age_in_months"].transform("mean")
test_use["age_max_by_district"] = test_use.groupby("district")["age_in_months"].transform("max")
test_use["age_min_by_district"] = test_use.groupby("district")["age_in_months"].transform("min")
test_use["age_std_by_district"] = test_use.groupby("district")["age_in_months"].transform("std")

test_use["age_mean_by_layout"] = test_use.groupby("layout")["age_in_months"].transform("mean")
test_use["age_max_by_layout"] = test_use.groupby("layout")["age_in_months"].transform("max")
test_use["age_min_by_layout"] = test_use.groupby("layout")["age_in_months"].transform("min")
test_use["age_std_by_layout"] = test_use.groupby("layout")["age_in_months"].transform("std")

test_use["age_mean_by_structure"] = test_use.groupby("structure")["age_in_months"].transform("mean")
test_use["age_max_by_structure"] = test_use.groupby("structure")["age_in_months"].transform("max")
test_use["age_min_by_structure"] = test_use.groupby("structure")["age_in_months"].transform("min")
test_use["age_std_by_structure"] = test_use.groupby("structure")["age_in_months"].transform("std")

test_use["building_floor_mean_by_sta"] = test_use.groupby("nearest_sta")["building_floor"].transform("mean")
test_use["building_floor_max_by_sta"] = test_use.groupby("nearest_sta")["building_floor"].transform("max")
test_use["building_floor_min_by_sta"] = test_use.groupby("nearest_sta")["building_floor"].transform("min")
test_use["building_floor_std_by_sta"] = test_use.groupby("nearest_sta")["building_floor"].transform("std")

test_use["building_floor_mean_by_district"] = test_use.groupby("district")["building_floor"].transform("mean")
test_use["building_floor_max_by_district"] = test_use.groupby("district")["building_floor"].transform("max")
test_use["building_floor_min_by_district"] = test_use.groupby("district")["building_floor"].transform("min")
test_use["building_floor_std_by_district"] = test_use.groupby("district")["building_floor"].transform("std")



# nan handling
for col in train_use.columns.values:
	train_use[col].fillna(-999, inplace=True)
	test_use[col].fillna(-999, inplace=True)


logger.debug(f"Using features:{train_use.columns.values}")

categorical_cols = ["district", "layout", "direction", "structure", "nearest_sta", "district_layout", 
                    "district_structure", "layout_structure"]



#################### 
## Train model
#################### 
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_use))
predictions = np.zeros(len(test_use))
feature_importance_df = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(folds.split(train_use, train_use["district"])):
    print(f"Fold {fold+1}")
    train_data = lgb.Dataset(train_use.iloc[train_idx], label=target[train_idx], categorical_feature=categorical_cols)
    val_data = lgb.Dataset(train_use.iloc[val_idx], label=target[val_idx], categorical_feature=categorical_cols)
    num_round = N_ROUNDS
    callbacks = [log_evaluation(logger, period=100)]
    clf = lgb.train(params, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=False, early_stopping_rounds=100, callbacks=callbacks)
    oof[val_idx] = clf.predict(train_use.values[val_idx], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = train_use.columns.values
    fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    feature_importance_df = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).head(50)
    logger.debug("##### feature importance #####")
    logger.debug(feature_importance_df)
    
    predictions += clf.predict(test_use, num_iteration=clf.best_iteration) / folds.n_splits


# post processing
post_process = pd.DataFrame()
post_process["pred"] = predictions
post_process["group"] = test_group

# trainの中に一致するものがあればそれにあわせる
# trainになかったものに対しても、testの予測値の平均をとる
pred_dic = post_process.groupby("group")["pred"].mean()
post_process["pred"] = post_process["group"].apply(lambda x: rent_dic[x] if x in rent_dic else pred_dic[x])
predictions = post_process["pred"]


cv_score = np.sqrt(mean_squared_error(oof, target))
logger.debug(f"5fold CV score: {cv_score}")
update_tracking(MODEL_ID, "cv_rmse", cv_score)



#################### 
## Submit
#################### 
spsbm = pd.read_csv("./data/sample_submit.csv", header=None)
spsbm.iloc[:, 1] = predictions
spsbm.to_csv(f"./submissions/{MODEL_ID}.csv", header=None, index=None)