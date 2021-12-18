import streamlit as st
import pickle
import pandas as pd
import numpy as np



def show_predict_page():
    st.title("Boston House Price Prediction")
    
    st.write("""### **Enter the details for Prediction**""")
    
    crim = st.number_input("Crime Rate", 0.0, 90.0, value=0.0)
    zn   = st.number_input("Zoning", 0.0, 100.0, value=0.0)
    indus = st.number_input("Industrial", 0.0, 30.0, value=0.0)
    chas = (
            "True", 
            "False"
            )
    chas = st.selectbox("Tracts with Charles River?", chas)
    nox = st.number_input("Nitric Oxide",0.0, 1.0, value=0.0)
    rm  = st.slider("Average Rooms", 0.0, 10.0, value=0.0)
    age = st.number_input("Age", 0.0, 100.0, value=0.0)
    dis = st.number_input("Distance", 0.0, 15.0, value=0.0)
    rad = st.number_input("Radius", 1.0, 25.0, value=1.0)
    tax = st.number_input("Tax", 0.0, 1000.0, value=0.0)
    ptratio = st.number_input("PTRatio", 0.0, 25.0, value=0.0)
    b = st.number_input("B", 0.0, 500.0, value=0.0)
    lstat = st.number_input("LSTAT", 0.0, 50.0, value=0.0)
    
    prediction = st.button("Predict the Median value of owner-occupied homes in $1000's")
    
    if prediction:
        if chas == True:
            chas = 1
        else:
            chas = 0
        
        # Convering Data from the user to the required format   
        df = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, ptratio, b, lstat]], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'B', 'LSTAT'])
        st.write("""### **Specified Input Parameters**""")
        st.table(df)
        # Loading the model for data transformation
        min_max_scaler = pickle.load(open('src/Models/MinMaxScaler.pkl', 'rb'))
        standard_scaler = pickle.load(open('src/Models/StandardScaler.pkl', 'rb'))
        
        # Applying the MinMaxscaler to the data 
        df[['B','ZN', 'CRIM']] = min_max_scaler.transform(df[['B','ZN', 'CRIM']])
        
        # Apply the Log Transformation to the data
        col_lst=['B','CRIM','ZN']
        for i in col_lst:
            df[i] = df[i].apply(lambda x: np.log(x+1))
        
        # Applying the Standard Scaler to the data
        col_lst = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','PTRATIO','B','LSTAT']
        df[col_lst] = standard_scaler.transform(df[col_lst])
        
        # Load the model for prediction
        final_model = pickle.load(open('src/Models/XGBRegressor.pkl', 'rb'))
        
        
        st.subheader(f'The Predicted Value is: {final_model.predict(df.values)[0]:.2f}')
        #
# if __name__ == '__main__':
#     show_predict_page()