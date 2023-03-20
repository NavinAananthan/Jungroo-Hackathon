import cv2
import numpy as np
import pytesseract
from translate import Translator
from PIL import Image, ImageDraw, ImageFont

img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img_bin = 255 - img_bin

kernel = np.ones((2, 1), np.uint8)
img_bin = cv2.erode(img_bin, kernel, iterations=1)
img_bin = cv2.dilate(img_bin, kernel, iterations=1)

text = pytesseract.image_to_string(img_bin, lang='eng')

translator= Translator(to_lang="hi")
translated_text = translator.translate(text)
print(translated_text)

img_bin[text != ' '] = 255

# Create a new image with the translated text
font_path = 'C:\Windows\FontsKruti_Dev_010.ttf'
img_pil = Image.fromarray(img_bin)
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype(font_path, size=20)
draw.text((50, 50), translated_text, font=font)
img_with_text = np.array(img_pil)

cv2.imshow('Original Image', img)
cv2.imshow('Translated Image', img_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
