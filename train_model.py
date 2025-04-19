import tensorflow as tf
import json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# Directory setup
train_dir = './train'
test_dir = './test'

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(  # Changed from train_datagen to test_datagen
    test_dir,  # Changed from train_dir to test_dir
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# CNN Model
model = load_model('skin_disease_model.h5') 

# Continue training
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)
# classification report

def evaluate_model_after_chunk(model, history, test_data):
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    # Classification report
    preds = model.predict(test_data)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_data.classes
    labels = list(test_data.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=labels))

# Save
model.save('skin_disease_model.h5')

evaluate_model_after_chunk(model, history, test_data)
