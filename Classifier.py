import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2

# Load the EMNIST dataset
emnist = fetch_openml('emnist', cache=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emnist.data, emnist.target, test_size=0.2)

# Train an SVM classifier on the training data
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Save the trained model as an XML file
svm_file = 'char_classifier.xml'
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(np.float32(X_train), cv2.ml.ROW_SAMPLE, y_train)
svm.save(svm_file)
