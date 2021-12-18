import numpy as np
import pandas as pd
from src.model_saver import model_save
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class preprocessing:
    """
    
    This module is used to preprocess the data with all the necessary scalers.
    
    """
            
    
    def __init__(self, data):
        self.data_scaled = data
        self.ms = model_save()
    
    def min_max_scaler(self,data,col_lst=['B','ZN','CRIM']):
        min_max = MinMaxScaler()
        data[col_lst] = min_max.fit_transform(data[['B','ZN','CRIM']])
        self.ms.save_model(min_max, 'MinMaxScaler')
        data = pd.DataFrame(data, columns=data.columns)
        return data
    
    def log_transformation(self,data,col_lst=['B','CRIM','ZN']):
        
        for i in col_lst:
            data[i] = data[i].apply(lambda x: np.log(x+1))
        return data
    
    def standard_scaler_transform(self,data,col_lst = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','PTRATIO','B','LSTAT']):
        
        scaler = StandardScaler()
        data[col_lst] = scaler.fit_transform(data[col_lst])
        self.ms.save_model(scaler, 'StandardScaler')
        return data
        
if __name__ == '__main__':
    pp = preprocessing()
    data_minmax        = pp.min_max_scaler(data=pp.data_scaled)
    data_log_transform = pp.log_transformation(data=data_minmax)
    data_scaler_transform = pp.standard_scaler_transform(data=data_log_transform)
    print(data_scaler_transform.head(15))