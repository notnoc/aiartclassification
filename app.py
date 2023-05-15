from flask import Flask, request, jsonify, render_template, session
import numpy as np
from PIL import Image
import tensorflow as tf
import base64

# Load the trained model
from keras.models import load_model
model = load_model('model_saved.h5', compile=False)

# Create the Flask app
app = Flask(__name__)
app.secret_key = 'my_secret_key'

# Define the prediction route
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']
        # Assume that the file input name is "image"
        #image_file = request.files['image']
        # Read the image data
        #image_data = image_file.read()
        # Convert the image data to a base64 string
        #encoded_image = base64.b64encode(image_data).decode('utf-8')
        # Store the base64 string in the session
        #session['image'] = encoded_image

        # Preprocess the image
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize to match model input shape
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make the prediction
        pred = model.predict(img)

        # Return the prediction result
        label = 'Fake' if pred[0] < 0.5 else 'Real'

        # Retrieve the base64 string from the session
        #encoded_image = session.get('image', None)
        #if encoded_image is not None:
        # Convert the base64 string back to binary image data
           #image_data = base64.b64decode(encoded_image)
        # Render the result template with the image data

        return render_template('result.html', label = label)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)