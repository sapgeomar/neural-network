# Sahed Ahmed Palash, Biological Oceanography, GEOMAR
# Deep Learning Basics
# Training the neural network using my own images

# import the necessary packages
import numpy as np
import pickle                                                                         # for data saving purposes
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt

# loading the train data and labels
pickle_in = open("trainData.pickle", "rb")
x = pickle.load(pickle_in)
pickle_in = open("trainLabel.pickle", "rb")
y = pickle.load(pickle_in)
#print(x[6],y[6])

# lets normalize the data before we feed them into the CNN based on the gray scale which is min 0 and max 250
x = x/255.0

# once scalling is done we are now ready to build our model (sequential)
model = Sequential()                                                                 # defining our model as sequential
model.add(Conv2D(128, (3,3), input_shape = x.shape[1:]))                              # adding the convolutional first layer with a 64 node and 3 by 3 size
model.add(Activation("relu"))                                                        # activation with rectified linear unit
model.add(MaxPool2D(pool_size=(2,2)))                                                # pooling with size of 2 by 2

model.add(Conv2D(128, (3,3)))                                                         # adding the convolutional second layer with a 64 node and 3 by 3 size
model.add(Activation("relu"))                                                        # activation with rectified linear unit
model.add(MaxPool2D(pool_size=(2,2)))                                                # pooling with size of 2 by 2

# now we have a 2 by 64 layer CNN and now we will add a final dense layer into the model
model.add(Flatten())                                                                 # we have to flatten the data before the dense layer
model.add(Dense(64))                                                                 # adding the dense layer with 64 nodes

# so, we have all the important three layers needed for a CNN model, now we will make the output layer
model.add(Dense(1))                                                                  # adding output dense layer with only one node
model.add(Activation("sigmoid"))                                                     # activation with sigmoid

# bingo, we have build our model precisely, now we have to complile our model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# lets run the fit into the model to be trained
model.fit(x,y, epochs=10)

# okay! our model is trained with almost 100 % accuracy, now lets make the prediction to see how model is working!!!
# loading the test data and labels
pickle_in = open("testData.pickle", "rb")
a= pickle.load(pickle_in)
pickle_in = open("testLabel.pickle", "rb")
b = pickle.load(pickle_in)

# defining the testlabel as fiha and sahed

def labels():
    for i in b:
        if i == 0:
            print("calanoids")
        else:
            print("eucalanoids")

#labels()

# prediction for the accurcy
predictions = model.predict(a)

# prediction for the max provability of sigmoid curve for exact labels
prediction_result = np.argmax(predictions[18])
print(prediction_result)

#  single image changing the dimension to visualize the the numpy array as an image
img = a[18]
img = (np.expand_dims(img,0))
plt.imshow(np.squeeze(img))
plt.xlabel(b[18])
plt.show()
