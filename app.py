from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
# from src.predict_page import show_predict_page
# from src.explore_page import show_explore_page
from src.training import training
from src.prediction import Prediction

app = Flask(__name__) # initializing a flask app

@app.route('/', methods = ['GET']) # route to display the home page):
@cross_origin()

def homePage():
    return render_template("index.html") 

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            crim =float(request.form['Crime Rate'])
            zn = float(request.form['Zoning'])
            indus = float(request.form['Industrial'])
            chas = request.form['Tracts with Charles River?']
            if(chas=='yes'):
                chas=1
            else:
                chas=0
            nox = float(request.form['Nitric Oxide'])
            rm = float(request.form['Average Rooms'])
            age = float(request.form['Age'])
            dis = float(request.form['Distance'])
            rad = float(request.form['Radius'])
            ptratio = float(request.form['PTRatio'])
            b = float(request.form['B'])
            lstat = float(request.form['LSTAT'])
            
            df = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, ptratio, b, lstat]], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'B', 'LSTAT'])
            
            pred = Prediction(df)
            score = pred.prediction(df)
            print(f'prediction is, {score[0]}.2f')
            # showing the prediction results in a UI
            return render_template('results.html',prediction=score[0])
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')
   
if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app


# btn = st.selectbox("Select the page", ["Prediction","Explore"])
# if btn == "Prediction":
#     show_predict_page()
# elif btn == "Explore":
#     show_explore_page()
    
# if __name__ == "__main__":
#     btn = st.selectbox("Select the page", ["Prediction","Explore"])
#     if btn == "Prediction":
#         show_predict_page()
#     elif btn == "Explore":
#         show_explore_page()
