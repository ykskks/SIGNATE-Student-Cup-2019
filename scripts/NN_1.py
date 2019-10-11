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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

sys.path.append(".")
from utils import update_tracking, log_evaluation, preprocess_df, TabularDataset, EarlyStopping


#################### 
## Changes
#################### 
# MODEL_ID = "TEINEI_17"
MODEL_ID = "NN_1"



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
NUM_EPOCH = 10000
BATCH_SIZE = 64

update_tracking(MODEL_ID, "num_epoch", NUM_EPOCH)
update_tracking(MODEL_ID, "batch_size", BATCH_SIZE)


class SigNet(nn.Module):

  def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
               output_size, emb_dropout, lin_layer_dropouts):

    """
    Parameters
    ----------

    emb_dims: List of two element tuples
      This list will contain a two element tuple for each
      categorical feature. The first element of a tuple will
      denote the number of unique values of the categorical
      feature. The second element will denote the embedding
      dimension to be used for that feature.

    no_of_cont: Integer
      The number of continuous features in the data.

    lin_layer_sizes: List of integers.
      The size of each linear layer. The length will be equal
      to the total number
      of linear layers in the network.

    output_size: Integer
      The size of the final output.

    emb_dropout: Float
      The dropout to be used after the embedding layers.

    lin_layer_dropouts: List of floats
      The dropouts to be used after each linear layer.
    """

    super().__init__()

    # Embedding layers
    self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                     for x, y in emb_dims])

    no_of_embs = sum([y for x, y in emb_dims])
    self.no_of_embs = no_of_embs
    self.no_of_cont = no_of_cont

    # Linear Layers
    first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                lin_layer_sizes[0])

    self.lin_layers =\
     nn.ModuleList([first_lin_layer] +\
          [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
           for i in range(len(lin_layer_sizes) - 1)])
    
    for lin_layer in self.lin_layers:
      nn.init.kaiming_normal_(lin_layer.weight.data)

    # Output Layer
    self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                  output_size)
    nn.init.kaiming_normal_(self.output_layer.weight.data)

    # Batch Norm Layers
    self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
    self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                    for size in lin_layer_sizes])

    # Dropout Layers
    self.emb_dropout_layer = nn.Dropout(emb_dropout)
    self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                  for size in lin_layer_dropouts])

  def forward(self, cont_data, cat_data):

    if self.no_of_embs != 0:
      x = [emb_layer(cat_data[:, i])
           for i,emb_layer in enumerate(self.emb_layers)]
      x = torch.cat(x, 1)
      x = self.emb_dropout_layer(x)

    if self.no_of_cont != 0:
      normalized_cont_data = self.first_bn_layer(cont_data)

      if self.no_of_embs != 0:
        x = torch.cat([x, normalized_cont_data], 1) 
      else:
        x = normalized_cont_data

    for lin_layer, dropout_layer, bn_layer in\
        zip(self.lin_layers, self.droput_layers, self.bn_layers):
      
      x = F.relu(lin_layer(x))
      x = bn_layer(x)
      x = dropout_layer(x)

    x = self.output_layer(x)

    return x


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

rent_dic = train_processed.groupby("group")["rent"].mean()


test_processed["ku"] = test_processed["location"].apply(lambda x: re.search("(?<=都).*?区", x).group())
test_group = test_processed["ku"] + test_processed["building_floor"].astype(str) \
                    + test_processed["age_in_months"].astype(str) + test_processed["area"].astype(str)

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


# nan handling
for col in train_use.columns.values:
	train_use[col].fillna(-1, inplace=True)
	test_use[col].fillna(-1, inplace=True)


# scaling 
categorical_cols = ["district", "layout", "direction", "structure"]
con_cols = [col for col in train_use.columns if col not in categorical_cols]

X_train, X_val, y_train, y_val = train_test_split(train_use, target_log, test_size=0.2, random_state=42)

sc = StandardScaler()
train_use = pd.concat([X_train[categorical_cols].reset_index(drop=False),\
                    pd.DataFrame(sc.fit_transform(X_train[con_cols]), columns=con_cols)], axis=1)
val_use = pd.concat([X_val[categorical_cols].reset_index(drop=False),\
                    pd.DataFrame(sc.transform(X_val[con_cols]), columns=con_cols)], axis=1)
test_use = pd.concat([test_use[categorical_cols],\
                    pd.DataFrame(sc.transform(test_use[con_cols]), columns=con_cols)], axis=1)


logger.debug(f"Using features:{train_use.columns.values}")



# perparing dataloader and stuff
train_use["target_log"] = y_train
val_use["target_log"] = y_val

train_dataset = TabularDataset(data=train_use, cat_cols=categorical_cols, output_col="target_log")
val_dataset = TabularDataset(data=val_use, cat_cols=categorical_cols, output_col="target_log")
test_dataset = TabularDataset(data=test_use, cat_cols=categorical_cols, output_col=None)

train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

cat_dims = [int(train_use[col].nunique()) for col in categorical_cols]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

print(cat_dims, emb_dims)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = SigNet(emb_dims, no_of_cont=len(con_cols), lin_layer_sizes=[50, 100],output_size=1,\
                emb_dropout=0.04,lin_layer_dropouts=[0.001,0.01]).to(device)


#################### 
## Train model
#################### 
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
early_stopping = EarlyStopping(patience=10, verbose=True)


for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times

    model.train() 
    train_loss = 0.0
    val_loss = 0.0
    for i, data in enumerate(train_dataset):
        # get the inputs; data is a list of [inputs, labels]
        y, cont_x, cat_x = data

        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        y  = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(cont_x, cat_x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()

    
    model.eval() 
    for data in val_dataset:
        y, cont_x, cat_x = data

        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        y  = y.to(device)

        # forward + backward + optimize
        outputs = model(cont_x, cat_x)
        loss = criterion(preds, y)

        val_loss += loss.item()


    print(f"epoch {epoch+1}: train_loss:{train_loss}, val_loss:{val_loss}")

    early_stopping(valid_loss, model)
        
    if early_stopping.early_stop:
        print("Early stopping")
        model.load_state_dict(torch.load('checkpoint.pt'))
        break

print('Finished Training')


model.eval() 
# val prediction
val_predictions = np.zeros(len(val_dataset))
val_target = np.zeros(len(val_dataset))
for i, data in enumerate(val_dataset):
    y, cont_x, cat_x = data

    cat_x = cat_x.to(device)
    cont_x = cont_x.to(device)

    # forward + backward + optimize
    outputs = model(cont_x, cat_x)
    val_predictions[i: i+BATCH_SIZE] = outputs
    val_target[i: i+BATCH_SIZE] = y

val_predictions = np.expm1(val_predictions)  
val_target = np.expm1(val_target)  


# test prediction
predictions = np.zeros(len(test_dataset))
for i, data in enumerate(test_dataset):
    _, cont_x, cat_x = data

    cat_x = cat_x.to(device)
    cont_x = cont_x.to(device)

    # forward + backward + optimize
    outputs = model(cont_x, cat_x)
    preds[i: i+BATCH_SIZE] = outputs
    
predictions = np.expm1(predictions)


# post processing
post_process = pd.DataFrame()
post_process["pred"] = predictions
post_process["group"] = test_group

# trainの中に一致するものがあればそれにあわせる
# trainになかったものに対しても、testの予測値の平均をとる
pred_dic = post_process.groupby("group")["pred"].mean()
post_process["pred"] = post_process["group"].apply(lambda x: rent_dic[x] if x in rent_dic else pred_dic[x])
predictions = post_process["pred"]


cv_score = np.sqrt(mean_squared_error(val_predictions, val_predictions))
logger.debug(f"5fold CV score: {cv_score}")
update_tracking(MODEL_ID, "cv_rmse", cv_score)



#################### 
## Submit
#################### 
spsbm = pd.read_csv("./data/sample_submit.csv", header=None)
spsbm.iloc[:, 1] = predictions
spsbm.to_csv(f"./submissions/{MODEL_ID}.csv", header=None, index=None)