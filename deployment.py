from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('mnist_cnn_model.h5')

def preprocess_image(image):
    """
    Preprocess the input image to the format required by the model.
    """
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)  # Convert to a NumPy array
    image = image / 255.0  # Normalize pixel values to between 0 and 1
    image = np.expand_dims(image, axis=-1)  # Add a channel dimension
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)
        return jsonify({'digit': int(predicted_digit)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
