from tensorflow.keras.layers import Input,GaussianNoise
#-------------Below 2 lines comment If ur  !using wandb------------------------------
import wandb 
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
#-------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = Sequential([
    GaussianNoise(0.1, input_shape=(224, 224, 3)),
    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),

    # layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(32, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



train_data_dir = './training_data_new_seg'
valid_data_dir = './validation_data_new_seg'

datagen = ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Rotate images by up to 20 degrees
    width_shift_range=0.2,  # Shift width by up to 20% of the image width
    height_shift_range=0.2,  # Shift height by up to 20% of the image height
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom in/out by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in any missing pixels with the nearest value
)

# Create data generators for training and validation data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Input image size
    batch_size=32,
    class_mode='categorical'
)

valid_generator = datagen.flow_from_directory(
    valid_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)



epochs = 100
history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator)
model.save('cnn_segmentation.h5')
