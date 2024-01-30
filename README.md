# webcam_digit_recognition

This repository contains 3 items (code for model, model itself, python program for input and output)

So starting with our model, i used mnist data for training purpose which consist of images size (28x28).
This is a simple Convolutional Neural Network (CNN) implemented using TensorFlow and Keras for image classification. The model is designed to work with grayscale images of size (28, 28, 1). It consists of convolutional layers, max-pooling layers, a flattening layer, and two dense layers.

Model Architecture->
Input Layer: Convolutional layer with 32 filters and a kernel size of (3, 3), using the ReLU activation function.
MaxPooling Layer: Pooling layer with a 2x2 filter to reduce spatial dimensions.
Flatten Layer: Flattens the input to a one-dimensional array to connect with dense layers.
Dense Layer (Hidden): Fully connected layer with 512 units and the ReLU activation function.
Dense Layer (Output): Fully connected layer with 10 units (output classes for classification) and the softmax activation function.

#python script

We capture input through our webcam and then resize it to 28x28, as our model is trained on this specific shape. To achieve this, we utilize the concept commonly referred to as a Region of Interest (ROI or a bounding box) . This involves cropping the center of our frame using a basic technique. After resizing the extracted region, we normalize the pixel values by dividing them by 255 before passing it into the model. 
