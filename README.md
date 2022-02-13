# Dataset Description
To build the model for skin cancer detection, we have used a dataset of 3297 images of benign and malignant skin images from kaggle. These images were provided in two directories from kaggle as train data and test data for training, and testing purpose. However, we have further split this train data as 80% training data and 20% as validation data. Figure 1 shows the distribution of overall dataset used for model training.

The original skin images were of size (224, 244, 3) that is for three channels red, green, and blue. All these images were resized to shape (32,32,3) and normalized between range 0 to 1 for faster processing using value 1./255. These resize and normalization operaions on images are performed using parameters passed thorough ImageDataGenerator from keras library.

<!-- <img src="https://github.com/snehalpimple/TaskScheduling/blob/main/Slide1.PNG" width=60% height=60%> -->
<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide2.PNG" width=60% height=60%>

# Convolutional Neural Network 

We use below CNN architecture to classify between malignant vs benign images. We start with a baseline architecture consisting of two convolution layers and one dense layer, then progressively vary the hyperparameters to see
their impact on the performance. 

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide3.PNG" width=60% height=60%>

# Hyperparameter selection

Below table 1 shows the experimentation performed and the results achieved in each trial until the best model was selected. The first parameter is the number of layers in the network. We observed that with the base model of 2 convolutional layers and 1 dense layer we achieved accuracy of 81.21%. When we increased one dense layer, we had increment of 0.46%. Further in experiment number 3, we added one maxpooling layer after first convolutional layer and there was gradual increment in the performance of the model and the accuracy was 82.27%.

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide4.PNG" width=60% height=60%>

Further we used another hyperparameter that is number of kernels. So far in the first 3 experiments we used 32 kernels in each convolutional layers. Moving forward we changed hyperparameter of number of kernels from 32 to 64, and as seen in table 1 experiment number 4, the test accuracy achieved was 83.03% .Next, we changed the number of kernels from 64 to 128 in experiment number 5, and the accuracy slightly decreased to 82.12% from 83.03%. We observed here that, increasing the number of kernels improved the accuracy, however up to certain numbers. Moving ahead for the best selection of model, we changed the kernel size from 3x3 to 5x5. And we observed that by changing the kernel size did not improve the accuracy. So, we learn here that it is not necessary always to make a model complex, because this can lead to overfitting of the model.

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide5.PNG" width=60% height=60%>

Later, we had multiple trials with different hyperparameters along with various combinations. Like, adding maxpooling and dropout layers between the convolutional layers. Dropout layer has been widely used to improve the performance of CNN. It helps to reduce the interdependent learning among the neurons to avoid overfitting [14]. Dropout is developed in such a way that it can forget some weights so that only important features of the data is stored. In this project, we added a dropout layer right before the fully-connected layer. Also, we kept different number of kernels for different convolutional layers. And finally, after tuning the hyperparameters to achieve the best model we finalized the model with total three convolutional layers of 64, 64 and 128 kernels of 3*3 size, followed by maxpooling and dropout. Along with this in fully connected we had two dense layers with 80 and 256 number of perceptrons. 

The weights are learned and models are trained using forward and backward propagation. We use gradient descent approach for model to learn weights. we have used Adam optimizer with learning rate of 0.01 and binary cross entropy loss. The binary cross entropy loss is calculated as;

After selecting the best hyperparameters for CNN model the training accuracy for skin cancer detection model is 82.80% and that of validation data is 79.40%. The final performance of this CNN model on test data is 84.70%.

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide6.PNG" width=60% height=60%>

Below figure 3a and 3b, shows the plots for accuracy and loss versus every epochs on training and validation data to observe the overfitting of the model.

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide7.PNG" width=60% height=60%>

# Transfer Learning
Transfer learning is a machine learning method where a model is developed
for a task A and we reuse learned features of task A to perform task B.
Transfer learning is useful when our current problem has limited dataset for
training our model.
For example, we developed a model for classification of cats and dogs image
detection. Here we have 25k images of cats and dogs, and we made the
hypothesis that features learned from cats and dogs images such as edge,
shapes, etc, can be reused to detect features of benign and malignant images
in the convolution layers for skin cancer detection CNN model.
In the current case of skin cancer detection the number of samples in dataset
is only 3297 images of benign and malignant . This is not sufficient to train
the model to its best. So, we use transfer learning method. In our case we use
the trained model of cats and dogs image classification to learn the features
first. In particularly while using transfer learning, we freeze the training
of initial convolutions layers and fine tuned other convolution layers during
backpropagation of model training.
Figure 4 shows the overall architecture of using transfer learning approach
for skin cancer detection, while using learned features from cats and dogs
image classification problem.

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide8.PNG" width=60% height=60%>

##Transfer Learning Results
Figure 5 shows the results achieved during performing CNN for cats and dogs
image classification problem. We used the same CNN model for cats and dogs
classification that was finally selected for skin cancer detection problem. The
knowledge about features are transferred to skin cancer detection problem
and the accuracy achieved is shown in same table.

We also observe that the execution time taken for Skin cancer detection using
transfer learning was much faster than building the Skin cancer detection
using CNN. Below table shows the execution time required for model training
for skin cancer detection using CNN and Transfer Learning. The platform
used while working on project was GPU (limited access) from Google Colab,
using Python.

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide9.PNG" width=60% height=60%>

# Conclusion

<img src="https://github.com/snehalpimple/skincancer_cnn_transferlearning/blob/main/PPT/Slide10.PNG" width=60% height=60%>
