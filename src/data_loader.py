
#Neccessary Imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')

boston = load_boston()

class data_loader():
    """
    
    This class is used to load the Boston dataset by default.
    
    """
    
    def __init__(self, data=boston):
        self.data = data
        X = pd.DataFrame(data.data, columns=boston.feature_names)  
        Y = pd.DataFrame(data.target, columns=['MEDV']) 
        self.bos = pd.concat([X, Y], axis=1)
        self.bos.drop(columns=['TAX'], inplace=True)
        #print(self.bos.head())
        
if __name__ == '__main__':
    dl = data_loader()
    print(dl.bos.head())