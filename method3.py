from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
from translate import Translator

# Load the image
img = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Binarize the image using Otsu's method
_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Invert the image
img_bin = 255 - img_bin

# Apply morphological operations to remove noise
kernel = np.ones((2, 1), np.uint8)
img_bin = cv2.erode(img_bin, kernel, iterations=1)
img_bin = cv2.dilate(img_bin, kernel, iterations=1)

# Extract text from the image
text = pytesseract.image_to_string(img_bin, lang='eng')

# Translate the text to Hindi
translator = Translator(to_lang="hi")
translated_text = translator.translate(text)

# Create a new image with the translated text
img_pil = Image.fromarray(img_bin)
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype("arial.ttf", 20)

# split the text into lines
lines = translated_text.split('\n')

# find the line containing the search string
line_index = 0
line_offset = 0
for i, line in enumerate(lines):
    if translated_text.strip() in line:
        line_index = i
        line_offset = line.index(translated_text.strip())
        break

# calculate the position and size of the original text
line_height = font.getsize(' ')[1]
original_text_position = (line_offset, line_index * line_height)
original_text_size = draw.textsize(translated_text, font=font)

# check the color of each pixel in the original text area
for x in range(original_text_position[0], original_text_position[0] + original_text_size[0]):
    for y in range(original_text_position[1], original_text_position[1] + original_text_size[1]):
        pixel_color = img_bin[y, x]
        if pixel_color == 0:
            img_bin[y, x] = 255

# draw the translated text
translated_text_position = (original_text_position[0], original_text_position[1] + original_text_size[1])
translated_text_size = draw.textsize(translated_text, font=font)
text_color = (255, 0, 0) # red color for text
draw.text(translated_text_position, translated_text, fill=text_color, font=font)

img_with_text = np.array(img_pil)

# Display the original and the modified images
cv2.imshow('Original Image', img)
cv2.imshow('Modified Image', img_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
