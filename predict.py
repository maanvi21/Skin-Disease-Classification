import numpy as np
import tensorflow as tf 
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import json

# Load model
import tensorflow as tf
model = tf.keras.models.load_model('skin_disease_model.h5')


# Load class labels
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse mapping from index to label
class_labels = {v: k for k, v in class_indices.items()}

def predict_image(image):
    # Convert to RGB just in case
    image = image.convert('RGB')
    
    # Resize to match model's input size
    image = image.resize((64, 64))
    
    # Convert image to array and normalize
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    # Get the class label
    predicted_label = class_labels[predicted_index]
    
    return predicted_label
