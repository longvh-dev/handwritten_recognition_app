import os

import numpy as np
from flask import Flask, jsonify, render_template, request

from predictors import predict_digit


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
model_path = 'save/model.pth'
model = CNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
# model = torch.load(model_path, map_location=torch.device('cuda'))
model.eval()


def predict_digit(input):
    with torch.no_grad():
        output = model(torch.tensor(input, dtype=torch.float32))
        probabilities = torch.nn.functional.softmax(output, dim=1)
    return probabilities.tolist()


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
