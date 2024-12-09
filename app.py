from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PerdictionPipeline
from src.logger import logging

# Initialize the Flask application
application = Flask(__name__)  # Creating the Flask app instance
app = application  # Assigning the Flask app instance to `app` for easier reference

# Default route for the application
@app.route('/')
def index():
    """
    Renders the main index page of the application.
    Typically used as the entry point or landing page.
    """
    return render_template('index.html')

# Route for handling prediction requests
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles GET and POST requests for predicting data points.

    - GET: Renders the home page where users can input data.
    - POST: Processes the form data submitted by the user, performs prediction, and displays results.
    """
    if request.method == 'GET':
        # Render the form for user input
        return render_template('home.html')
    else:
        # Process the submitted form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        # Convert the form data into a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        logging.info("Prediction started")

        # Initialize the prediction pipeline and make predictions
        predict_pipeline = PerdictionPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info("Prediction completed")

        # Render the results on the home page
        return render_template('home.html', results=results[0])

# Entry point for running the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
