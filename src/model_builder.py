import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


class model_building:
    
    def __init__(self,data):
        self.final_data = data
        self.X = self.final_data.drop(['MEDV'], axis=1)
        self.Y = self.final_data['MEDV']
        self.scores_df = pd.DataFrame()
    
    def split_data(self,X,Y,test_size=0.2,random_state=42):
        """
        This method is used to split the data into train and test sets.
        """
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=test_size, random_state=random_state)
        X.to_csv('EDA\\X_test_data.csv',index=False)
        return X_train, X_test, Y_train, Y_test
    
    def model_builder(self,model_object, x_train, x_test, y_train, y_test):
        model = model_object
        model.fit(x_train, y_train)
        cv = cross_val_score(model, x_train, y_train, cv=5)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        r2_train_score = model.score(x_train, y_train)
        r2_test_score = model.score(x_test, y_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print(f"**********************Model : {model.__class__.__name__}**********************")
        print("CV: ", cv.mean())
        print('R2_score (train): ', r2_train_score)
        print('R2_score (test): ', r2_test_score)
        print("MSE (train): ", mse_train)
        print("MSE (test): ", mse_test)
        scores_df_temp = pd.DataFrame({'model' : [model.__class__.__name__] ,'R2_train': [r2_train_score], 'R2_test': [r2_test_score], 'MSE_train': [mse_train], 'MSE_test': [mse_test], 'CV_scores': [cv.mean()]},)
        self.scores_df = pd.concat([self.scores_df, scores_df_temp], ignore_index=True)
        self.scores_df.to_csv('src/scores/model_scores.csv',index=False)
        return model, model.__class__.__name__, mse_test, r2_train_score
    
    def find_best_model(self,all_model_mse_scores):
        best_model = min(all_model_mse_scores, key=all_model_mse_scores.get)
        best_mse = all_model_mse_scores[best_model]
        best_r2 = all_model_mse_scores[best_model][1]
        print('Best model: ', best_model)
        print('Best MSE: ', best_mse)
        print('Best R2: ', best_r2)
        return best_model,best_mse,best_r2
    
    