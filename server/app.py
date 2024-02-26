import os

import numpy as np
from flask import Flask, jsonify, render_template, request

from predictors import predict_digit


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
model_path = '../save/model.pth'


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    input = ((255 - np.array(request.json, dtype = np.uint8)) / 255.0).reshape(
        1, 1, 28, 28)

    result = predict_digit(input, model_path)

    return jsonify({ 'result': result })


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True, port = 5000)
