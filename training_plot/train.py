#import the required libraries
import keras
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from trainingplot import TrainingPlot

# Provide the output path and name for the plot
filename = 'output/training_plot.jpg'

# Create an instance of the TrainingPlot class with the filename.
plot_losses = TrainingPlot()

# Find the number of classes
num_classes = 10

# Split the data into train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train_cat = np.squeeze(keras.utils.to_categorical(y_train, num_classes))
y_test_cat = np.squeeze(keras.utils.to_categorical(y_test, num_classes))

#Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

#Compile the model
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

#Train the model using callback to the TrainingPlot class object
model.fit(x_train, y_train_cat,
         epochs=25,
         validation_data=(x_test, y_test_cat),
         callbacks=[plot_losses])
