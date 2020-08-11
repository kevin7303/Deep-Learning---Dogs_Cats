# Dogs_Cats_CNN
Basic Convolutional Neural Network for Binary Image Classification

# Dogs and Cats Convolutional Neural Network - Project Overview 
* Creating a basic Convolutional Neural Network to classify images of dogs and cats.
* Dataset taken from Kaggle competition Dogs vs Cats with 25,000 training images.
* Created an Image generator input images in batches in order to reduce memory usage.
* Used Rectified Linear Unit Activation Function transformation to capture Non-Linearity 
* Used Sigmoid Activation Function to capture Binary Classification on Output Layer
 

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, keras, matplotlib
**Original Kaggle Dataset:** https://www.kaggle.com/c/dogs-vs-cats/data

## Data Overview
* 25,000 Training Images that will be split into Train and Validation sets (80/20 split)
* 12,500 images of dogs
* 12,500 images of cats
* Varied dimensions and sizes
* Backgrounds included and non blurred

## Image Generator and Transformations

rotation_range=15,
rescale=1./255,
shear_range=0.1,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1,
validation_split=0.2


## Model Building 
# Convolutional Neural Network Architecture
Input Layer
Convolutional Layer - ReLU Activation

Second Convolutional Layer - ReLU Activation
Batch Normalization
MaxPooling - Size 2,2
Dropout Layer with 25% Drop

Third Convolutional Layer - ReLU Activation
Batch Normalization
MaxPooling - Size 2,2
Dropout Layer with 25% Drop

Flattening Layer
Dense (Fully Connected) Layer - ReLU Activation
Batch Normalization
Dropout Layer with 50% Drop
Output Layer - Dense (Fully Connected) with 1 Output - Sigmoid Activation 

# CNN Compile
Loss - Binary CrossEntropy
Metric - Accuracy
Optimizer - adam

# Fitting Parameters
Epochs = 50
Callback - EarlyStop - Patience = 10
Batch Size = 16

## Results
![alt text](https://github.com/kevin7303/Deep-Learning---Dogs_Cats/blob/master/Accuracy%20Graph.png "Loss and Accuracy - Train vs Validation")


* Highest Validation Accuracy  0.9825721383094788
* Last Epoch Validation Accuracy  0.9443109035491943
__________________________________________________________
* Highest Training Accuracy  0.8757500052452087
* Last Epoch Training Accuracy  0.8726500272750854
