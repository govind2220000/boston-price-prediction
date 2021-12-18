# import numpy as np
# import pandas as pd
# import pickle

# x_test = pd.read_csv('EDA\\Boston_test_data.csv')
# s = pd.DataFrame([[1.3, 9]], columns = ["A", "B"])
# print(s)
  
# # makes index continuous
# t = pd.DataFrame()
# t = t.append(s, ignore_index = True)  
# print(t)
  
# # Resultant data frame is of type float and float
# print(t.dtypes) 

# with open('src\Models\XGBRegressor.pkl', 'rb') as f:
#     pred = pickle.load(f)

# p = pred.predict([[-0.467815,0.427865,-1.287909,0.0,-0.144217,0.413672,-0.120013,0.140214,-0.982843,-0.666608,0.408414,-1.075562]])
# print(p)


import pandas as pd
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  
Y = pd.DataFrame(boston.target, columns=['MEDV']) 
bos = pd.concat([X, Y], axis=1)

for i in bos.columns:
    print(i, bos[i].min(),bos[i].max())


#-0.523001
# TAX       -0.666608
# PTRATIO   -0.857929
# B          0.396763
# LSTAT     -0.506457
# print(p)