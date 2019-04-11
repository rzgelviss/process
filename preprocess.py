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







