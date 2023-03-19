import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the EMNIST dataset
ds_train, ds_test = tfds.load('emnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

# Preprocess the data
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    label = tf.one_hot(label, depth=62)
    return image, label

ds_train = ds_train.map(preprocess_image)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(preprocess_image)
ds_test = ds_test.batch(32)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Define the model architecture
model = keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(62, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(ds_train, epochs=10, validation_data=ds_test)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(ds_test)
print('Test accuracy:', test_acc)

# Save the model
model.save('char_classifier.h5')
