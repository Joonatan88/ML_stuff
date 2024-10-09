import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Load data. x = pixels (hand written), y = actual number

x_train = tf.keras.utils.normalize(x_train, axis=1) #Normalize the pixels to scale down
x_test = tf.keras.utils.normalize(x_test, axis=1)

y_train = to_categorical(y_train, 10)  #Hot-one vectors
y_test = to_categorical(y_test, 10)

model = tf.keras.models.Sequential() #initialize neuralnetwork
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(64, input_dim = 784, activation="sigmoid")) #Hidden layers
model.add(tf.keras.layers.Dense(32, input_dim = 784, activation="sigmoid"))
model.add(tf.keras.layers.Dense(16, input_dim = 784, activation="sigmoid"))
model.add(tf.keras.layers.Dense(10, activation="sigmoid")) #output layer

#keras.optimizers.SGD(learning_rate=0.3)    #learning rate used for sgd

#optimizer updates weights with learning rate to minimize loss (sgd = stochastic gradient decent)
#loss calculates average squared difference between predicted and target values (mse = mean squared error)
#metrics are used to evaluate the performance of the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  #adam optimizer and gets better results for me than sgd 

epochs = 20
tr_hist = model.fit(x_train, y_train, epochs=epochs, verbose=1) #train
plt.plot(tr_hist.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training'], loc='upper right')
plt.show()

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=1) #evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")