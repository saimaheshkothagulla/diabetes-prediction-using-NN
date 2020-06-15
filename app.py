import numpy as np
import pandas as pd
from flask import Flask, request, render_template
app = Flask(__name__)

dataset = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\flaskapp\\preprosseddata.csv')
from keras.models import load_model

dataset_X = dataset.iloc[:,[1,2,3,4, 5,6, 7,8]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    model1 = load_model('C:\\Users\\Lenovo\\Desktop\\flaskapp\\model.h5')
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model1.predict(sc.transform(final_features))
    pred = round(prediction[0][0])
    if pred == 1:
        pred = "**You have Diabetes, please consult a Doctor."
    elif pred == 0:
        pred = "**You don't have Diabetes."
    output=pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
