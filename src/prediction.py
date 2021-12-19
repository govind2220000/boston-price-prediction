import pickle
import pandas as pd
import numpy as np

class Prediction:
    
    def __init__(self,data) -> None:
        self.data = data
    
    def prediction(self,data):
        
        # Loading the model for data transformation
        min_max_scaler = pickle.load(open('src/Models/MinMaxScaler.pkl', 'rb'))
        standard_scaler = pickle.load(open('src/Models/StandardScaler.pkl', 'rb'))
        
        # Applying the MinMaxscaler to the data 
        data[['B','ZN', 'CRIM']] = min_max_scaler.transform(data[['B','ZN', 'CRIM']])
        
        # Apply the Log Transformation to the data
        col_lst=['B','CRIM','ZN']
        for i in col_lst:
            data[i] = data[i].apply(lambda x: np.log(x+1))
        
        # Applying the Standard Scaler to the data
        col_lst = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','PTRATIO','B','LSTAT']
        data[col_lst] = standard_scaler.transform(data[col_lst])
        
        # Load the model for prediction
        final_model = pickle.load(open('src/Models/XGBRegressor.pkl', 'rb'))
        score = final_model.predict(data.values)
        
        return score