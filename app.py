import streamlit as st
from src.predict_page import show_predict_page
from src.explore_page import show_explore_page


btn = st.selectbox("Select the page", ["Prediction","Explore"])
if btn == "Prediction":
    show_predict_page()
elif btn == "Explore":
    show_explore_page()
    
# if __name__ == "__main__":
#     btn = st.selectbox("Select the page", ["Prediction","Explore"])
#     if btn == "Prediction":
#         show_predict_page()
#     elif btn == "Explore":
#         show_explore_page()
