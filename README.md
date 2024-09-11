# Chicken Disease Prediction App

This Flask-based web application predicts whether a chicken is healthy or suffering from a disease based on an image of its fecal matter. The app uses a machine learning model trained on image data to classify the images into various disease categories.

## Features
- **Prediction Classes**: The model classifies the images into the following categories:
  - `cocci`
  - `healthy`
  - `ncd`
  - `pcrcocci`
  - `pcrhealthy`
  - `pcrncd`
  - `pcrsalmo`
  - `salmo`

- **Model Performance**:
  - **Loss**: `0.5047`
  - **Accuracy**: `94.78%`
  - **Precision**: `95.12%`
  - **Recall**: `94.45%`

## Dataset
The data for training the model was sourced from **Kaggle**, consisting of labeled images related to different chicken diseases and healthy samples. Various **image augmentation techniques** were applied during training to improve the model's generalization and robustness.

## Technologies Used
- **Flask**: For web application and API development.
- **Machine Learning**: Convolutional Neural Network (CNN) for image classification.
- **Keras/TensorFlow**: For building and training the model.
- **HTML/CSS/JavaScript**: For creating the front-end interface.

## How the App Works
1. **Image Input**: Users can upload an image of chicken feces through the interface.
   - Option to either upload from the device or take a picture directly using the camera.
2. **Prediction**: Once the image is uploaded, the app predicts the class (disease) of the chicken and displays the result along with a confidence score.
3. **Results**: The result will show the predicted disease or health status of the chicken, aiding in early detection of potential health issues.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone <https://github.com/Shadowwalker10/chicken-disease-classification>
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Flask app:
   python app.py
4. Access the app in your browser at http://localhost:8080.

## Usage Instructions
- **Upload an image**: Click on "Choose File" to upload an image or "Take Picture" to use your device’s camera.
- **Prediction**: Click the "Predict" button to get the result.
- **View Result**: The app will display the predicted disease or health status with the confidence percentage.

## Model Evaluation Metrics
The model was evaluated based on the following metrics:
- **Accuracy**: `94.78%` - The overall percentage of correct predictions.
- **Precision**: `95.12%` - The proportion of true positives among all predicted positives.
- **Recall**: `94.45%` - The proportion of true positives among all actual positives.

## Future Improvements
- Add more diseases for prediction.
- Enhance model performance with more data and advanced architectures.
- Provide detailed explanations of the predictions for user understanding.

