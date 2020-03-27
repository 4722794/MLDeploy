#Importing necessary packages
import numpy as np
from flask import Flask, request, render_template
import pickle
from fastai.tabular import *
import os

#Saving the working directory and model directory
cwd = os.getcwd()
path = cwd + '/model'


#Initializing the FLASK API
app = Flask(__name__)

#Loading the saved model using fastai's load_learner method
model = load_learner(path, 'export.pkl')

#Defining the home page for the web service
@app.route('/')
def home():
    return render_template('myindex.html')

#Writing api for inference using the loaded model
@app.route('/predict',methods=['POST'])

#Defining the predict method get input from the html page and to predict using the trained model

def predict():
    
    try:
    	#all the input labels . We had only trained the model using these selected features.
        
        labels = ['age', 'sex', 'cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']

        #Collecting values from the html form and converting into respective types as expected by the model
        Age =  int(request.form["age"])
        Sex =  request.form["sex"]
        Cough = request.form["cough"]
        Fever =  request.form["fever"]
        Chills = request.form["chills"]
        Sore_throat = request.form["sore_throat"]
        Headache =  request.form["headache"]
        Fatigue = request.form["fatigue"]


        # Age =  22
        # Sex =  'male'
        # Cough = 'Yes'
        # Fever =  'Yes'
        # Chills = 'Yes'
        # Sore_throat = 'No'
        # Headache =  'No'
        # Fatigue = 'Yes'
        #making a list of the collected features
        features = [Age, Sex, Cough, Fever,Chills, Sore_throat, Headache, Fatigue]

        #fastai predicts from a pandas series. so converting the list to a series
        to_predict = pd.Series(features, index = labels)

        #Getting the prediction from the model and rounding the float into 2 decimal places
        prediction = int(round(float(model.predict(to_predict)[1]),0))

        
        # Making all predictions below 0 lakhs and above 200 lakhs as invalid
        if features[2:] == ['No','No','No','No','No','No']:
            return render_template('myindex.html', prediction_text= f"You don't show any discernable symptoms")
        elif prediction != 0:
            return render_template('myindex.html', prediction_text= f'Please wait for {prediction} days before you see a medical professional')
        else:
            return render_template('myindex.html', prediction_text='You may need immediate assistance')

    except Exception as e:
        return render_template('myindex.html', prediction_text= f'your input {features} is invalid')

if __name__ == "__main__":
    app.run(debug=True)