#loading the libraries
from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle
import os

#initialising the flask
app = Flask(__name__)


#loading the models
ct1 = pickle.load(open('col_trans1.pkl','rb'))
ct2 = pickle.load(open('col_trans2.pkl','rb'))
model = pickle.load(open('random_forest_classifier.pkl','rb'))

#loading the dataset to show the set of valid inputs to the user
df = pd.read_csv('df_reduced.csv')
dl = {}

dl['FL_NUM'] = sorted(df.FL_NUM.unique())


@app.route('/')
def f1():
    return render_template("home.html",dl=dl)

@app.route('/prediction',methods = ['post'])
def f2():
    if request.method == 'POST':
        results = request.form
        response = {}
        dic = {}

        for key,value in results.items():
            dic[key] = [value]

        delay = int(dic['DEP_TIME'][0]) - int(dic['CRS_DEP_TIME'][0])
        dic['DEP_DELAY'] = [delay]
        dic['DEP_DEL15'] = [float(delay > 15)]

        df = pd.DataFrame(dic)

        df.FL_NUM = df.FL_NUM.astype('int')
        df.MONTH = df.MONTH.astype('int')
        df.DAY_OF_MONTH = df.DAY_OF_MONTH.astype('int')
        df.DAY_OF_WEEK = df.DAY_OF_WEEK.astype('int')
        df.CRS_ARR_TIME = df.CRS_ARR_TIME.astype('int')
        df.DEP_DELAY = df.DEP_DELAY.astype('float')
        df.DEP_DEL15 = df.DEP_DEL15.astype('float')

        if(dl['FL_NUM'].count(df['FL_NUM'][0]) == 0):
            response['result'] = "<span class='neg'>Enter the correct Flight number...</span>"
            return jsonify(response)


        x = df[['FL_NUM','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','ORIGIN','DEST','CRS_ARR_TIME','DEP_DEL15','DEP_DELAY']]
        print(x)

        x = pd.DataFrame(ct1.transform(x),columns=ct1.get_feature_names_out())
        x = pd.DataFrame(ct2.transform(x),columns=ct2.get_feature_names_out())


        y_p = model.predict(x)
        if(y_p):
            response['result'] = "<span class='neg'>Oww! , Flight-"+str(df.FL_NUM[0])+" may be delayed</span>"
        else:
            response['result'] = "<span class='pos'>Be Happy! , Flight-"+str(df.FL_NUM[0])+" will be on-time</span>"

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)