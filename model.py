import tensorflow as tf
import json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

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
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_data,
    validation_data=test_data,  # Changed from test_data to validation_data
    epochs=2
)

# Save
model.save('skin_disease_model.h5')

# Save class labels mapping
with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)