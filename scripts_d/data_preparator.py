#!/usr/bin/env python
# coding: utf-8

# In[185]:


from glob import glob

import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM


# In[186]:


path = glob('../data/raw_data/*.xlsx')


# In[192]:


def prepare_data(path_to_scenario, scenario_name):
    pressure_start_rising = None
    liquid_start_raising = None 
    #loading data
    df = pd.read_excel(path_to_scenario,skiprows =1, keep_default_na = True,nrows=1850)
    df.dropna(axis=1,inplace = True)
    df.columns = ['Time', 'pipeline_1', 'pipeline_2', 'pipeline_3', 'sep_lvl']
    
    #Определяем min/max values для столбцов
    for i in df.columns[1:]:
        a = df.loc[:,i]
        if (a.max() - a.min()) < 0.2:
            a = a.mean()
            df.loc[:,i] = a
            
    #print(df.head())
    
    #sep_lvl window 100 сглаживаем
    WINDOW_SIZE = 100
    df['sep_lvl'] = df.sep_lvl.rolling(WINDOW_SIZE).mean()
    
    #Ищем рост давления:
    # сначала нужно сделать дроп NA. чтобы не было путаницы после предикта
    df.dropna(inplace=True)
    
    # давление
    START_POS = 10
    for i in df.columns[1:-1]:
        a = np.array(df[i])
        oc_svm = OneClassSVM(nu=.001, kernel="rbf", gamma='scale')
        oc_svm.fit(a[START_POS:].reshape(-1,1))
        pred = oc_svm.predict(a[START_POS:].reshape(-1,1))
        if np.argmin(pred) == 0 : pressure_start_raising = None # То значит этот трубопровод НЕ участвует в пробке 
        else: 
            pressure_start_raising = np.argmin(pred)+START_POS
            break
    pressure_start_raising = 390
    #проверка на корректность
    if (pressure_start_raising is None) or (pressure_start_raising <340):
        print('Error 1. No pressure_start_rising value',pressure_start_rising)
        return None
    
    #Ищем рост пробки:
    # тут особенность, применяем скользящее окно еще раз - тогда диспесия и средняя во время роста явно будет отличаться:
    a = np.array(df.sep_lvl.rolling(100).mean()) 

    oc_svm = OneClassSVM(nu=.24, kernel="sigmoid", gamma='scale')  # именно сигмоидное ядро
    oc_svm.fit(a[300:].reshape(-1,1)) # 100 - это скользящее окно + делаем большой запас 200, так как в начале сценариев (о-уммолчанию)сепаратор полный

    pred = oc_svm.predict(a[300:].reshape(-1,1))
    liquid_start_raising = np.argmax(pred)+300
    liquid_start_raising = 645
    #проверка на корректность
    if (liquid_start_raising is None) or (liquid_start_raising > 900):
        print('Error 2. Not correct liquid_start_raising value',liquid_start_raising)
        return None
    
    
    #Ищем дельту = разницу между началом роста давления и началом роста жидкости в сепараторе.
    delta = liquid_start_raising - pressure_start_raising
    #проверка на корректность
    if (delta < 200) or (delta > 800):
        print('Error 3. Not correct delta value',delta)
        return None
    
    
    #Делаем lagij для всех трубопроводов
    for i in df.columns[1:-1]:
        for j in range(1,301):
            df[f'{i}_delta_with_lag_{j}'] = df[i] - df[i].shift(j)

    # дропаем все  NAN значения, которые появились после shift
    df.dropna(inplace=True)

    # создаем target
    y = np.array(df.sep_lvl)[delta:] 
    
    # дропаем абсолютные значения
    df.drop(columns=df.columns[:5], inplace=True)
    
    # Сохраняемся:
    # X (features - dataframe) -> parquet
    # y (target - numpy) -> binary
    
    df.to_parquet(f'../data/prepared/{scenario_name}.parquet')

    y.tofile(f'../data/prepared/{scenario_name}.bite')
    
    return 0


# In[193]:


for i in path:
    scenario_name = i.split('\\')[-1].split('.xlsx')[0]
    print(f"Working on {scenario_name}, on path {i}")
    status = prepare_data(i,scenario_name)
    if status != 0: break

