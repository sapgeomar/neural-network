# Sahed Ahmed Palash, Biological Oceanography, GEOMAR
# Deep Learning Basics,
# Loading images for test data and labels

# import the necessary packages
import cv2                                                                            # to deal with image operations
import numpy as np                                                                    # to deal with array operations
import matplotlib.pyplot as plt                                                       # to visualize the data
import os                                                                             # to iterate through the directory
from os.path import join                                                              # to join the path
import random                                                                         # for shuffling the images
import pickle                                                                         # for data saving purposes

# specify the data directory and category folders
datadir = "/media/sahed/Windows 10/ML/images/test_images"                            # path of the image folder
categories = ["calanoids", "eucalanoids", "euphausiids", "oithona", "pleuroncodes"]  # name of the folder

# iterate over the images
for category in categories:                                                           # interating over the folders
    path = os.path.join(datadir, category)                                            # giving the path
    for img in os.listdir(path):                                                      # iterating over the images
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)         # array of image from col to gray scale
        plt.imshow(img_array, cmap="gray")                                            # to show the image
        #plt.show()
        #break                                                                        # break the loop after 1st image
    #break                                                                            # break the loop after 1st directory

# now lets print the picture as an array (bunch of numercial numbers) and its shape (y and x pixel numbers)
#print(img_array)
#print(img_array.shape)

# lets fix the pixel numbers as uniformed for better future training because different picture have different shape
img_size = 50                                                                         # define the number of shape 50/50
new_array = cv2.resize(img_array, (img_size, img_size))                               # resize the old array to 50/50
#plt.imshow(new_array, cmap="gray")                                                   # visualize the image
#plt.show()                                                                           # see the image
#print(new_array)                                                                     # see the new array
#print(new_array.shape)                                                               # see the new shape

# lets create the test datasets
test_data = []                                                                        # creating an empty list
def create_test_data():                                                               # defining a function with new array
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)                                        # defining the class number
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            test_data.append([new_array, class_num])                                  # append the empyty list with new array and class number
create_test_data()                                                                    # calling the function
#print(len(test_data))                                                                # see the length of the training data
#random.shuffle(test_data)                                                            # give training data a shuffle
#lets see the training data how does it looks like
#for sample in training_data[:10]:                                                    # iterate over the first ten training data
    #print(sample[0], [1])                                                            # print the first one

# lets pack the features and labels into two different variables to feed them into our neural network
x = []                                                                                # create an empty list for features
y = []                                                                                # create an empty list for the labels
for features, labels in test_data:                                                    # defining and calling image array and category as features and labels
    x.append(features)                                                                # adding features in x
    y.append(labels)                                                                  # adding labels in y

# we cannot feed this variables into our neural network directly (x has to be a numpy array), we have to transformm them (x,y)
x = np.array(x).reshape(-1, img_size, img_size, 1)                                    # creating as numpy array with reshaping

# once our training data is ready we will store this data into our same directory by using pickle package (x and y variables)
pickle_out = open("testData.pickle", "wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("testLabel.pickle", "wb")
pickle.dump(y,pickle_out)
pickle_out.close()

