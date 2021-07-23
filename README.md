# Using KNN with digit recognition

For this exercise, I use the KNN algorithm to classify images of digits.

There are 3000 training examples in digit_dataset.csv, where the first column is the label and the rest of the training example are 28 pixels by 28 pixels grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity atthat location. The 28 by 28 grid of pixels is "unrolled" into a 784-dimensionalvector. Each of these training examples becomes a single row in our data matrix X. This results in a 3000 by 784 matrix X where every row is a training example for a handwritten digit image.

This was completed for Machine Learning COSC-A406 at Loyola University New Orleans in Spring 2021.