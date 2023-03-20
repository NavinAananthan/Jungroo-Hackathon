import numpy as np
import cv2
from translator import Translator
import tensorflow as tf

# Read the image in grayscale
img = cv2.imread('input.png', 0)

# Binarize the image
_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Invert the image
img_bin = 255 - img_bin

# Apply morphological operations to remove noise
kernel = np.ones((2, 1), np.uint8)
img_bin = cv2.erode(img_bin, kernel, iterations=1)
img_bin = cv2.dilate(img_bin, kernel, iterations=1)

# Find contours in the image
contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Load the character classifier (replace this with your own classifier)
char_classifier = tf.keras.models.load_model('char_classifier.h5')

# Create a new image with the same size and color as the original image
translated_img = np.zeros_like(img)

# Initialize the translator
translator = Translator(to_lang="Hindi")

# Sort the contours from left to right and top to bottom
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
(contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda x: (x[1][1], x[1][0])))

# Iterate over each contour and extract the text
translated_text = ''
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the region of interest (ROI) from the image
    roi = img[y:y + h, x:x + w]

    # Resize the ROI to a fixed size
    roi = cv2.resize(roi, (20, 20))

    # Flatten the ROI into a 1D array
    roi_flat = roi.reshape((1, 400)).astype(np.float32)

    # Normalize the ROI
    roi_norm = roi_flat / 255.0

    # Predict the character using the classifier
    char_code = np.argmax(char_classifier.predict(roi_norm))

    # Convert the character code to ASCII
    char = chr(char_code)

    # Translate the character to Hindi
    translated_char = translator.translate(char)

    # Add the translated character to the translated text
    translated_text += translated_char

    # Draw the translated text on the new image
    cv2.putText(translated_img, translated_char, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)

# Print the translated text
print(translated_text)

# Save the translated image
cv2.imwrite('translated.png', translated_img)
