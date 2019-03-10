import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def cal_woe_iv(df):
    df_copy = df.copy()
    woe_IV_List = []
    rate =df_copy['class'].sum()/(df_copy['class'].count()-df_copy['class'].sum())
    colList = df_copy.columns[:-1]
    for col in colList:
        if df_copy[col].dtypes == np.dtype('O'):
            group = df_copy[col]
        else:
            try:
                group = pd.qcut(df_copy[col],4)
            except:
                group = pd.cut(df_copy[col],4)
        grouped = df_copy['class'].groupby(group,as_index=True).value_counts()
        woe=np.log(grouped.unstack().iloc[:,1]/grouped.unstack().iloc[:,0]/rate)
        IV=((grouped.unstack().iloc[:,1]/df_copy["class"].sum()-grouped.unstack().iloc[:,0]/(df_copy["class"].count()-df_copy["class"].sum()))*woe).sum()
        woe_IV_List.append([col,woe,IV])
    return woe_IV_List

def feature_selection(woe_iv_list):
    feature = []
    for i in range(len(woe_iv_list)):
        if woe_iv_list[i][2] >= 0.025:
            feature.append({'col':woe_iv_list[i][0],'iv':woe_iv_list[i][2]})
    return pd.DataFrame(feature)
	
def vif_cal(df,list):
	data_x = df[list.col.values].select_dtypes(include=['float64'])
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(data_x.values, i) for i in range(data_x.shape[1])]
	vif["features"] = data_x.columns
	return vif

def replace_woe(df):
    df_new = pd.DataFrame()
    df_copy = df.copy()
    rate =df_copy['class'].sum()/(df_copy['class'].count()-df_copy['class'].sum())
    colList = df_copy.columns[:-1]
    for col in colList:
        if df_copy[col].dtypes == np.dtype('O'):
            group = df_copy[col]
        else:
            try:
                group = pd.qcut(df_copy[col],4)
            except:
                group = pd.cut(df_copy[col],4)
        grouped = df_copy['class'].groupby(group,as_index=True).value_counts()
        woe=np.log(grouped.unstack().iloc[:,1]/grouped.unstack().iloc[:,0]/rate)
        for i in group.unique():
            group.replace(i,woe[i],inplace=True)
            df_new[col] = group
    df_new['class'] = df_copy['class']
    return df_new
