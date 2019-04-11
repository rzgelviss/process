import pandas as pd
import numpy as np
import warnings
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,KFold,ShuffleSplit,ParameterGrid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# %matplotlib inline


path = r"D:/Ren/wxwj/耐蚀钢/焊接工艺规程.xlsx"
def read_file(path):
    df = pd.read_excel(path, encoding='utf-8')
    return df

#删除空值
def drop_na(df):
    th_high = 0.85
    th_low = 0.02
    df.dorpna(axis=1, thresh=(1-th_high)*df.shape[0], inplace=True)
    return df

#空值填充
def fill_na(df, column):
    df = df.loc[df[pd.isnull(df['column'])].index,['column']] = 1
    return df

#异常值检测
def outliers(df):
    outliers_threshold = 2
    li_col = [col for col in df]
    for col in li_col:
        label = df[col].value_counts(dropna=False).index.tolist()
        for i ,num in enumerate(df[col].value_counts(dropna=False).values):
            if num <= outliers_threshold:
                df.loc[df[col] == label[i], [col]] = np.NaN
    return df

#单一类别特征占比大于threshold特征删除
def single_feature(df):
    single_threshold = 0.98
    li_col = [col for col in df]
    for col in li_col:
        rate = df[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > single_threshold:
            print('类别{0}占比{1}'.format(col,rate))
            df.drop(col, axis=1, inplace=True)
    return df

#特征占比图
def show_feature_rate(df,col):
    cnt = df[col].value_counts(dropna=False).sort_index(ascending=True)
    plt.figure(figsize=(10, 5))
    cnt.plot.bar()
    plt.title(col)
    plt.show()

#共线特征删除
def correlation(df):
    li_col = [col for col in df.columns]
    corr_matrix = df.corr()
    correlation_threshold = 0.98
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])
    for column in to_drop:
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
        col_values = list(upper[column][upper[]])











