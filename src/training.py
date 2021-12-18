from src.data_loader import data_loader
from src.model_builder import model_building
from src.data_preprocessing import preprocessing
from src.model_saver import model_save
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

class training:
    
    def train(self):
        
        
        #Initializing ModelSaver
        ms = model_save()
        
        #Initializing DataLoader
        dl = data_loader()

        #Initializing DataPreprocessing
        pp = preprocessing(data = dl.bos)
        data_minmax        = pp.min_max_scaler(data=pp.data_scaled)
        data_log_transform = pp.log_transformation(data=data_minmax)
        data_scaler_transform = pp.standard_scaler_transform(data=data_log_transform)

        ##Initializing ModelBuilder
        mb = model_building(data = data_scaler_transform)
        x_train,x_test,y_train,y_test = mb.split_data(mb.X,mb.Y,test_size=0.2)

        scores_model = dict() # This dictionary is used to store the scores of all the models so that best model can be selected with least MSE

        # Model 1: Linear Regression
        lr = LinearRegression()
        model_lr, model_name ,model_lr_mse, model_lr_r2 = mb.model_builder(lr, x_train, x_test, y_train, y_test)
        scores_model[model_name] = model_lr_mse, model_lr_r2

        # Model 2: Decision Tree Regressor
        dtc = DecisionTreeRegressor(criterion='mse', splitter='best' ,max_depth=4)
        model_dtc, model_name, model_dtc_mse, model_dtc_r2 = mb.model_builder(dtc, x_train, x_test, y_train, y_test)
        scores_model[model_name] = model_dtc_mse, model_dtc_r2

        # Model 3: Random Forest Regressor
        rfc = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=4)
        model_rfc, model_name, model_rfc_mse, model_rfc_r2 = mb.model_builder(rfc, x_train, x_test, y_train, y_test)
        scores_model[model_name] = model_rfc_mse, model_rfc_r2

        # Model 4: XGBoost Regressor
        xgb = XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=100)
        model_xgb, model_name, model_xgb_mse,  model_xgb_r2 = mb.model_builder(xgb, x_train, x_test, y_train, y_test)
        scores_model[model_name] = model_xgb_mse,model_xgb_r2

        # Finding the best model with least MSE
        best_model,best_mse,best_r2 = mb.find_best_model(scores_model)

        #list of all the models
        model_dict = {f'{model_lr.__class__.__name__}': model_lr, f'{model_dtc.__class__.__name__}':model_dtc, f'{model_xgb.__class__.__name__}':model_xgb, f'{model_rfc.__class__.__name__}':model_rfc}

        #Dumping the best model
        for i in model_dict:
            if i == best_model:
                ms.save_model(model_dict[i],best_model)