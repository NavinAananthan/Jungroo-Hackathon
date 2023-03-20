# Jungroo-Hackathon

## Method1

 * This file classifier uses TensorFlow Datasets to load the EMNIST dataset, preprocesses the data, defines a convolutional neural network (CNN) model, trains the model, evaluates it on a test set, and saves the model.
 * The EMNIST dataset consists of images of handwritten characters from 62 classes (10 digits, 26 uppercase letters, and 26 lowercase letters). The tfds.load function is used to load the 'emnist' dataset, with the 'balanced' split, which contains an equal number of examples for each class. The train_test_split function from scikit-learn is used to split the training data into training and validation sets.
 * The preprocess function is defined to normalize the pixel values of the images, add a channel dimension, and one-hot encode the labels. The map function is used to apply the preprocess function to each element of the datasets, and the batch function is used to batch the data into batches of 32.
 * The CNN model architecture consists of two convolutional layers followed by max pooling layers, a flatten layer, and two dense layers. The model is compiled with the Adam optimizer and categorical cross-entropy loss, and trained for 10 epochs with the fit function.
 * Finally, the model is evaluated on the test set using the evaluate function and the test accuracy is printed. The model is saved as a Keras .h5 file using the save method of the Sequential class.
 * The given code is used to extract text from an image and translate it from English to Hindi using Image processing techniques and Translation API. Firstly, the image is read in grayscale and then binarized using thresholding techniques. The image is then inverted and morphological operations are applied to remove noise. Contours are then found in the binarized image to identify the regions of text. A pre-trained character classifier is loaded to classify each character in the image. The contours are sorted from left to right and top to bottom to ensure that the characters are processed in the correct order. The text is then extracted from each contour by resizing, flattening, normalizing, and predicting the character using the classifier. The extracted character is then translated to Hindi using the translation tool. The translated text is then concatenated to form the complete translated text. The translated text is then placed back onto the image using OpenCV's putText function. Finally, the translated image is saved to disk. This code can be used to process scanned documents or images with text in English and translate them to Hindi.



## Method2
 * This code reads an image ('img.png') using OpenCV and converts it to grayscale. Then it applies Otsu's thresholding to binarize the image, followed by morphological operations to remove noise. The text in the binarized image is extracted using Tesseract OCR library. The extracted text is then translated to Hindi using the translate library.
 * The translated text is then superimposed on the binarized image using the Kruti Dev font, and the resulting image is displayed using OpenCV. The original image is also displayed alongside for comparison.
 * Overall, the code performs OCR and translation of text in an image, and displays the translated text overlaid on the original image.


## Method3
 * This method is same as Method 1 but added with additional image processing techniques such as skewness with the borders to have a accurate image classification
