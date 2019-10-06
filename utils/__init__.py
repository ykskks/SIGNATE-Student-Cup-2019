import logging
import re

import pandas as pd
import numpy as np
from lightgbm.callback import _format_eval_result 


def update_tracking(model_id, field, value, csv_file='logs/history.csv',
                    integer=False, digits=None):
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except:
        df = pd.DataFrame()

    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[model_id, field] = value # Model number is index
    df.to_csv(csv_file)


def log_evaluation(logger, period=100, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback


def preprocess_df(df):
    ### location ###
    df["district"] = df["location"].apply(lambda x: re.search("(?<=都)(.*?)(?=区)", x).group())
    
    ### access ###
    df["access"] = df["access"].apply(lambda x: re.sub("\(.*?\)", "", x)) 
    df["access"] = df["access"].apply(lambda x: re.sub("\（.*?\）", "", x))
    df["access"] = df["access"].str.replace(" ", "")
    df["access"] = df["access"].str.replace("　", "")
    df["access"] = df["access"].str.replace("・", "")
    df["access"] = df["access"].str.replace("：", "")
    df["access"] = df["access"].str.replace("「", "")
    df["access"] = df["access"].str.replace("」", "")
    df["access"] = df["access"].str.replace("『", "")
    df["access"] = df["access"].str.replace("』", "")
    df["access"] = df["access"].str.split("\t\t").apply(lambda lis: "\t\t".join([sta for sta in lis if "バス" not in sta])) # バスはとりあえず除く
    df["access"] = df["access"].str.replace("\t", "")
    df["access"] = df["access"].str.replace("線", "線\t")
    df["access"] = df["access"].str.replace("ライン", "ライン\t")
    df["access"] = df["access"].str.replace("ライナー", "ライナー\t")
    df["access"] = df["access"].str.replace("エクスプレス", "エクスプレス\t")
    df["access"] = df["access"].str.replace("総武線\t・中央線", "総武中央線")
    df["access"] = df["access"].str.replace("駅", "駅\t")
    df["access"] = df["access"].str.replace("分", "分\t\t")
    df["access"] = df["access"].apply(lambda x: re.sub("分\t\t$", "分", x)) # 行末の処理
    df["access"] = df["access"].apply(lambda x: re.sub("駅\t$", "駅", x)) # 行末の処理

    df["access_line"] = df["access"].str.split("\t\t").apply(lambda lis: [sta.split("\t")[0] for sta in lis])
    df["access_line"] = df["access_line"].apply(lambda lis: [line for line in lis if "線" in line or 
                                                                    "ライナー" in line or
                                                                    "ライン" in line or 
                                                                    "エクスプレス" in line])

    df["access_sta"] = df["access"].str.split("\t\t").apply(lambda lis: [sta.split("\t")[1] for sta in lis 
                                                                    if len(sta.split("\t")) > 1 and "分" not in sta.split("\t")[1]])
    df["access_sta"] = df["access_sta"].apply(lambda lis: [sta for sta in lis if "駅" in sta])

    df["access_min"] = df["access"].str.split("\t\t").apply(lambda lis: [sta.split("\t")[2] for sta in lis    
                                                                    if len(sta.split("\t")) > 2 and "分" in sta.split("\t")[2]])
    df["access_min"] = df["access_min"].apply(lambda lis: [int(re.search(r"\d+(?=分)", mins).group()) for mins in lis])

    access_line = pd.get_dummies(df['access_line'].apply(pd.Series).stack()).groupby(level=0).apply(sum)
    df = pd.concat([df, access_line], axis=1)

    access_sta = pd.get_dummies(df['access_sta'].apply(pd.Series).stack()).groupby(level=0).apply(sum)
    df = pd.concat([df, access_sta], axis=1)


    ### layout ###
    df["is_K"] = df["layout"].str.contains("K").astype(int)
    df["is_R"] = df["layout"].str.contains("R").astype(int)
    df["is_L"] = df["layout"].str.contains("L").astype(int)
    df["is_D"] = df["layout"].str.contains("D").astype(int)
    df["is_S"] = df["layout"].str.contains("S").astype(int)
    df["num_room"] = df["layout"].apply(lambda x: re.match(r"\d", x).group()).astype(int)


    ### age ### 
    df["age"] = df["age"].replace("新築", "0年0ヶ月") # 0年0ヶ月の使用物件はほぼないので新築と同じと判断
    df["age_year"] = df["age"].apply(lambda x: re.search("[0-9]*(?=年)", x).group()).astype(int)
    df["age_month"] = df["age"].apply(lambda x: re.search("[0-9]*(?=ヶ月)", x).group()).astype(int)
    df["age_in_months"] = df["age_year"] * 12 + df["age_month"]


    ### direction ### 
    df["direction"].fillna("方向なし", inplace=True) # 欠損してる住宅は古いのが多いなど、特徴として扱う
    df["has_N"] = df["direction"].str.contains("北").astype(int)
    df["has_S"] = df["direction"].str.contains("南").astype(int)
    df["has_E"] = df["direction"].str.contains("東").astype(int)
    df["has_W"] = df["direction"].str.contains("西").astype(int)


    ### area ###
    df["area"] = df["area"].str[:-2].astype(float)


    ### floor ### 
    # testにnan一つあり
    def get_building_floor(floor):
        if pd.isnull(floor):
            return np.nan
        else:
            if re.search(r"\d+(?=階建)", floor):
                return int(re.search(r"\d+(?=階建)", floor).group())
            else:
                return np.nan

    def get_room_floor(floor):
        if pd.isnull(floor):
            return np.nan
        else:
            floor = floor.replace("階建", "")
            if re.search(r"\d+(?=階)", floor):
                return int(re.search(r"\d+(?=階)", floor).group())
            else:
                return np.nan

    df["building_floor"] = df["floor"].apply(get_building_floor)
    df["room_floor"] = df["floor"].apply(get_room_floor)

    df["has_underground"] = df["floor"].apply(lambda x: int("地下" in x) if not pd.isnull(x) else np.nan)


    ### bath_toilet ### 
    df["bath_toilet"] = df["bath_toilet"].str.replace("\t／\t", "／\t")

    bath_toilet = df["bath_toilet"].str.get_dummies(sep="／\t")
    bath_toilet[df["bath_toilet"].isnull()] = np.nan

    df = pd.concat([df, bath_toilet], axis=1)
   

    ### kitchen ### 
    df["kitchen"] = df["kitchen"].str.replace("\t／\t", "／\t")

    kitchen = df["kitchen"].str.get_dummies(sep="／\t")
    kitchen[df["kitchen"].isnull()] = np.nan

    df = pd.concat([df, kitchen], axis=1)
   

    ### broad_com ### 
    df["broad_com"] = df["broad_com"].str.replace("\t／\t", "／\t")

    broad_com = df["broad_com"].str.get_dummies(sep="／\t")
    broad_com[df["broad_com"].isnull()] = np.nan

    df = pd.concat([df, broad_com], axis=1)


    ### facility ### 
    df["facility"] = df["facility"].str.replace("／\t", "\t")

    facility = df["facility"].str.get_dummies(sep="\t")
    facility[df["facility"].isnull()] = np.nan

    df = pd.concat([df, facility], axis=1)


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

    df["bicycle_parking"] = df["parking"].apply(bicycle_parking)
    df["car_parking"] = df["parking"].apply(car_parking)
    df["bike_parking"] = df["parking"].apply(bike_parking)


    ### environment ###
    def clean_environment(environment):
        if pd.isnull(environment):
            return np.nan
        else:
        
            return re.findall("(?<=【).*?(?=】)", environment)

    df["environment_list"] = df["environment"].apply(clean_environment)
    
    env_uniq = []
    for lis in df["environment_list"].dropna():
        for i in lis:
            env_uniq.append(i)
            
    env_uniq = list(set(env_uniq))

    for col in env_uniq:
        df[col] = 0
        df[col] = df["environment"].apply(lambda x: int(re.search(f"(?<=【{col}】 ).*?(?=m)", x).group()) \
                                                    if (not pd.isnull(x) and re.search(f"(?<=【{col}】 ).*?(?=m)", x)) else 0)
  
    df.loc[df["environment"].isnull(), env_uniq] = np.nan # nanのところはゼロではなくnan
        

    ### structure ### 


    ### contract_period ### 
    def is_fixed_term(period):
        if pd.isnull(period):
            return np.nan
        elif "定期借家" in period:
            return 1
        else:
            return 0 

    df["fixed_term"] = df["contract_period"].apply(is_fixed_term)


    def get_contract_period_year(period):
        if pd.isnull(period):
            return np.nan
        elif re.search(r"\d+(?=年間)", period):
            return int(re.search(r"\d+(?=年間)", period).group())
        elif "年" in period and "ヶ月間" in period:
            return int(re.search(r"\d+(?=年)", period).group())
        elif "まで" in period: # 契約期間 (2019年8月が起点となります)
            if int(re.search(r"\d+(?=月)", period).group()) >= 8:
                return int(re.search(r"\d+(?=年)", period).group()) - 2019
            else:
                return int(re.search(r"\d+(?=年)", period).group()) - 2019 - 1
        else:
            return 0

    def get_contract_period_month(period):
        if pd.isnull(period):
            return np.nan
        elif "ヶ月間" in period:
            return int(re.search(r"\d+(?=ヶ月間)", period).group())
        elif "まで" in period: # 契約期間 (2019年8月が起点となります)
            if int(re.search(r"\d+(?=月)", period).group()) >= 8:
                return int(re.search(r"\d+(?=月)", period).group()) - 8
            else:
                return int(re.search(r"\d+(?=月)", period).group()) + (12 - 8)
        else:
            return 0

    df["contract_period_year"] = df["contract_period"].apply(get_contract_period_year)
    df["contract_period_month"] = df["contract_period"].apply(get_contract_period_month)
    df["contract_period_in_months"] = df["contract_period_year"] * 12 + df["contract_period_month"]

    
    return df