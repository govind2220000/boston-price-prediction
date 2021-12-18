import streamlit as st
import pandas as pd
from PIL import Image
def show_explore_page():
    img1 = Image.open("EDA/Boston_corr.jpeg",)
    st.write("""### **Correlation between the features**""")
    st.image(img1,caption='''From the plot we can see that LSTAT HAVE HIGHLY NEGATIVE CORRELATION WITH TARGET VALUE MEDV.
            From the plot we can see that RM HAVE HIGHLY POSITIVE CORRELATION WITH TARGET VALUE MEDV.
            From the plot we can see that LSTAT HAVE NEGATIVE CORRELATION WITH TARGET VALUE MEDV.
            From the plot we can see that TAX IS HIGHLY CORRELATED WITH RAD (so we can drop any one feature from this.)''',use_column_width=True)
    
    img2 = Image.open("EDA/Boston_density.jpeg",)
    st.write("""### **Density Plot**""")
    st.image(img2,caption='''From the above plot we can see that the data distribution is highly scattered in B, CRIM, 
             and ZN column so we need to reduce that lets cross check it by regplot.''',use_column_width=True)
    
    img3 = Image.open("EDA/Boston_regplot.jpeg",)
    st.write("""### **Regplot**""")
    st.image(img3,caption='''From the above plot we can see that the data distribution in all columns''',use_column_width=True)
    
    img4 = Image.open("EDA/Boston_regression_prediction_plot_XGBRegressor.jpeg",)
    st.write("""### ** Best Model Regression Prediction Plot**""")
    st.image(img4,caption='''Prediction Plot for XGBRegressor Model''',use_column_width=True)
    
    df = pd.read_csv('src/scores/model_scores.csv')
    st.write("""### **Model Scores**""")
    st.write(df)
    