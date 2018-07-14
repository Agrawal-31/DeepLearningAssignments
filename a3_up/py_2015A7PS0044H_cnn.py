import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

K.set_image_dim_ordering('th')


#We are loading the data suing the keras dataset itself 
(Image_train, Label_train), (Image_test, Label_test) = mnist.load_data()


# For convlutional networks to be used in keras, the image of both the training and the testing 
# need to be of 4 dimensions.
# in our particular problem the dimensions are  [number of samples][number of pixel/depth][width][height]

no_samples_train = Image_train.shape[0]
no_samples_test = Image_test.shape[0]
width = Image_train.shape[1]
height = Image_train.shape[2]
no_pixels = 1 #this is because the images are in grey scale, if it were RGB it would have been 3

#number_categories = Label_test.shape[1]
number_categories = 10

#The reason why it is converted to float is so that we can normalize the values later into the range 0-1
Image_train = Image_train.reshape(no_samples_train, no_pixels, width, height).astype('float')
Image_test = Image_test.reshape(no_samples_test, no_pixels, width, height).astype('float')


#We are initialised the seed to some arbitrary value
intial_seed_value = 12
np.random.seed(intial_seed_value)

#As mentioned in the assignment the values in the images must be normalised to the range 0-1

denominator = 255

Image_train = Image_train/denominator
Image_test = Image_test/denominator


# creating the CNN model
model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape=(1, 28, 28), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(32, (5, 5), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
model.add(Dense(number_categories, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#MNIST dataset has only images of 10 numbers from 0-9
#the assingment asks for cross extropy error  function for this we need the one hot encoding for the labels
# one hot encode outputs
Label_train = np_utils.to_categorical(Label_train)
Label_test = np_utils.to_categorical(Label_test)

sep = int(0.8 * no_samples_train)
model.fit(Image_train[:sep], Label_train[:sep], validation_data=(Image_train[sep:], Label_train[sep:]), epochs=50, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(Image_test, Label_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
