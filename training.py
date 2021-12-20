import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  
Y = pd.DataFrame(boston.target, columns=['MEDV']) 

#Scaling the data
scaler = StandardScaler()
# Drop the unnecessary columns
X.drop(['TAX'], axis=1, inplace=True)
X[['CRIM','ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'B'  ]] = scaler.fit_transform(X[['CRIM','ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'B'  ]])

#Splitting in train and test
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)


lr = LinearRegression()

lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)


print(f'R2: {lr.score(X_test, Y_test)}')
print(f'RMSE: {mean_squared_error(Y_test, y_pred)}')

with open(f'{lr.__class__.__name__}.pkl', 'wb') as f:
    pickle.dump(lr, f)
    
with open(f'{scaler.__class__.__name__}.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# scaler_model = pickle.load(open(f'{scaler.__class__.__name__}.pkl', 'rb')) 
# model = pickle.load(open(f'{lr.__class__.__name__}.pkl', 'rb'))
# score = model.predict([[-0.467815,0.427865,-1.287909,0.0,-0.144217,0.413672,-0.120013,0.140214,-0.982843,-0.666608,0.408414,-1.075562]])
# print(f'{score[0]}')