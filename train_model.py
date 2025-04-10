import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
model.fit(
    train_data,
    validation_data=test_data,
    epochs=3  # Train for 3 more epochs
)

# Save
model.save('skin_disease_model.h5')