# Handwritten Digit Recognition App

## Description:

This project implements a web application using Flask and PyTorch to predict the digit drawn by users on a canvas element. It provides users with a visually intuitive way to interact with a machine learning model for digit recognition.

## Key Features:
- User-friendly canvas for drawing: Users can easily draw digits on a designated canvas area within the web interface.
- Real-time prediction: As users draw, the app dynamically sends the canvas data to the Flask server for prediction.
- Probability distribution: The predicted digit is displayed along with its corresponding probability distribution, 
  offering insights into the model's confidence level.
- Clear instructions: The application provides clear instructions to guide users on how to interact with the canvas 
  and interpret the results.

## Technologies Used:
- Flask: A lightweight web framework for building web applications in Python.
- PyTorch: A popular deep learning framework with GPU acceleration capabilities.
- HTML/CSS/JavaScript: Used for creating the interactive canvas interface and displaying predictions.

## Requirements:
Python 3.8 or higher
PyTorch 1.7.1 or higher

## Installation:

1. Clone the repository:

```
git clone https://github.com/longvh-dev/handwritten_recognition_app.git
```

2. Install dependencies (Using conda enviroment)

```
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```
Remove the <env_name> and replace it with your desired environment name.


3. Usage:

Start the Flask application:

```
python app.py
```

Open your web browser and navigate to http://127.0.0.1:5000/ (or the specified port in app.py).

Draw a digit on the canvas: Use your mouse or touch screen to create a digit within the designated area.

Wait for the prediction: Once you stop drawing, the application will send the canvas data to the server. This triggers the prediction and displays the predicted digit and its probability distribution on the page.

## Trained Model:

The project use a CNN model with 2 layer for digit recognition. If you want to using different model, you can change 
the model in the `train/models/*.py` file and update the `app.py` file to load the new model.

## Contribution:

Contributions are welcome! Feel free to fork the repository, make changes, and create pull requests to improve the project.

## License:

This project is licensed under the MIT License. For more information, please refer to the LICENSE file.