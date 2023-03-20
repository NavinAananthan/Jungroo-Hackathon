import numpy as np
import cv2
from translator import Translator
import tensorflow as tf

# Read the image
img = cv2.imread('img.png')

# Apply Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to binarize the image
img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to remove noise
kernel = np.ones((2, 1), np.uint8)
img_morph = cv2.erode(img_thresh, kernel, iterations=1)
img_morph = cv2.dilate(img_morph, kernel, iterations=1)

# Find contours in the image
contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    # Filter out contours that are unlikely to be characters
    x, y, w, h = cv2.boundingRect(contour)
    if w < 10 or h < 10 or w > 100 or h > 100:
        continue
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.1 or aspect_ratio > 10:
        continue

    # Extract the region of interest (ROI) from the image
    roi = img_gray[y:y + h, x:x + w]

    moments = cv2.moments(roi)
    if abs(moments['mu02']) < 1e-2:
        skew = 0
    else:
        skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
    roi = cv2.warpAffine(roi, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    # Resize the ROI to a fixed size
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # Reshape the ROI to a 4D tensor to feed it into the model
    roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)

    # Predict the class of the character
    pred = char_classifier.predict(roi)

    # Translate the character to Hindi
    char = chr(np.argmax(pred) + 65)
    translated_char = translator.translate(char)
    translated_text += translated_char

    # Draw a bounding box around the character
    cv2.rectangle(translated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put the translated character on the image
    cv2.putText(translated_img, translated_char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# Print the translated text
print(translated_text)

# Save the translated image
cv2.imwrite('translated.png', translated_img)
