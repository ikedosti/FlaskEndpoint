"""
Waste Classifier API

This Flask web application provides a RESTful API for predicting the category of waste items using a pre-trained TensorFlow model.
The model classifies waste items into one of the following categories: "cardboard", "glass", "metal", "paper", "plastic", or "trash".



Dependencies:
- Flask
- TensorFlow
- NumPy
- Pillow (PIL)
- io

Usage:
1. Ensure all required dependencies are installed.
2. Run this script.
3. Send a POST request to the '/predict' endpoint with an image file attached.

Example using curl:
    curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:80/predict

Endpoints:
- /predict: Accepts POST requests with an attached image file for waste item classification.

"""

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained TenorFlow model
model = r'C:\Users\User\Downloads\flaskendpoint\1\1'
mod = tf.saved_model.load(model)

# Define waste item categories
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Define input image shape
im_shape = (256, 256)

def preprocess_image(img_buffer):
    """
    Preprocesses the input image.

    Args:
        img_buffer (bytes): Raw image data in bytes.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    img = Image.open(BytesIO(img_buffer))
    img = img.resize((im_shape[0], im_shape[1]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

def predict_image(img_array):
    """
    Makes predictions on the preprocessed image using the loaded TensorFlow model.

    Args:
        img_array (numpy.ndarray): Preprocessed image as a NumPy array.

    Returns:
        tuple: Predicted waste category and confidence score.
    """
    predictions = mod(img_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)], 100 * np.max(score)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the '/predict' endpoint.

    Returns:
        JSON: JSON response containing the predicted waste category and confidence score.
    """
    try:
        # Read image data from the request
        img_buffer = request.files['image'].read()
        print(f"Received request for an image of buffer length {len(img_buffer)}")

        # Preprocess the image
        img_array = preprocess_image(img_buffer)

        # Make predictions
        prediction, confidence = predict_image(img_array)

        # Prepare and return the result as JSON
        result = {
            'class': prediction,
            'confidence': confidence
        }

        return jsonify(result)

    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 and port 80
    app.run(host="0.0.0.0", port=80)

