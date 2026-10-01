import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_class_imbalance(file_path: 'str') -> tuple:
    file = pd.read_csv(file_path)
    
    l = sum(file['Irrigation_Need'] == 'Low')
    m = sum(file['Irrigation_Need'] == 'Medium')
    h = sum(file['Irrigation_Need'] == 'High')

    total = l + m + h

    print(f"Proportion of low: {l/total * 100}%")
    print(f"Propostion of high: {h/total * 100}%")
    print(f"Proportion of Medium: {m/total * 100}%")

    return (l, m, h)

def preprocessing(file_path: str, test: bool = False) -> tuple:

    file = pd.read_csv(file_path)

    # Threshold features matching the known generative formula
    file['soil_lt_25']  = (file['Soil_Moisture']  < 25).astype(float)
    file['temp_gt_30']  = (file['Temperature_C']  > 30).astype(float)
    file['rain_lt_300'] = (file['Rainfall_mm']    < 300).astype(float)
    file['wind_gt_10']  = (file['Wind_Speed_kmh'] > 10).astype(float)

    categorical_cols = file.select_dtypes(include=['object']).columns.to_list()

    for predictor in categorical_cols:
        categories = file[predictor].unique().tolist()
        for c in categories:
            file[predictor + '_' + c] = file[predictor] == c
    
    file = file.drop(categorical_cols, axis=1)

    file = file.drop('id', axis=1)
    
    if test:
        return file
    
    X = file.drop(['Irrigation_Need_Low', 'Irrigation_Need_Medium', 'Irrigation_Need_High'], axis=1)

    Y = file[['Irrigation_Need_Low', 'Irrigation_Need_Medium', 'Irrigation_Need_High']]

    return (X, Y)


