import numpy as np
import tensorflow as tf
import sys


def class_acc(pred, gt): #function to evaluate accuracy for models
   
    pred = np.array(pred)
    gt = np.array(gt)

    how_many_right = np.sum(pred == gt)
    accuracy = how_many_right / len(gt) * 100

    return accuracy

def log_likelihood(x, mean, var): #calculates the log likelihood
    return  np.sum(-0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var), axis=1)


if sys.argv[1] == "original":
    mnist = tf.keras.datasets.mnist
elif sys.argv[1] == "fashion":
    mnist = tf.keras.datasets.fashion_mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data() #load the data

x_train = x_train.reshape(x_train.shape[0], 784) #Training data is reshaped to 60000 x 784 matrix where one sample is shape of 1x784
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = tf.keras.utils.normalize(x_train, axis=1) #Normalize the pixels to scale down
x_test = tf.keras.utils.normalize(x_test, axis=1)

noise = np.random.normal(loc=0.0, scale=0.01, size=x_train.shape)
x_train = x_train + noise

mean_vector = np.zeros((10, 784)) #initialize mean and var vectors
variance_vector = np.zeros((10, 784))

#calculate mean and variance vectors
for k in range(10): 
    classdata = x_train[y_train == k] #classdata contains all training samples that match to k
    mean_vector[k] = np.mean(classdata, axis = 0) #computes the mean for every class
    variance_vector[k] = np.var(classdata, axis=0)  #computes the variance for every class. 


log_likelihoods = np.zeros((x_test.shape[0], 10)) #initialize likelihood array

for k in range(10):
    log_likelihoods[:, k] = log_likelihood(x_test, mean_vector[k], variance_vector[k]) #compute loglikelihoods for all test samples when compared to k

predict = np.argmax(log_likelihoods, axis=1) #picks out highest likelihoods
accuracy = class_acc(predict, y_test) #function to test how accurate our model is
print(f"Classification accuracy is: {accuracy:.2f}%")