import keras 
import numpy as np
from parser import load_data

training_data = load_data('data/training')
validation_data = load_data('data/validation')

model = Sequential()
model.add(Convolution2D(32,3,3 input_shape = (img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#classic CNN archtecture
#Input ->Conv->ReLU->Conv->ReLU->Pool->ReLU->Conv->RelU->Pool->FullyCOnnected
#ReLU is used to reduce image to non-negative value only- flattens the image to grayscale 
#Conv is a filter to convolute a matrix to the input matrix
#Max pooling takes largest no. from the widow 

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#now we need to apply dropout to prevent overfitting
model.add(Flatten()) 
model.add(Dense(64)) #initialise a fully connected layer
model.add(Activataion('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))#initialise another fully connected layer
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy' ,optimizer='rmsprop', metrics=['accuracy']) #for backpropagation of error

model.fit_generator(
        training_data,
        samples_per_epoch=2048,
        nb_epoch=30,
        validation_data=validation_data,
        nb_val_samples=832)

model.save_weights('models/simple_CNN.h5')

#testing the model
# img = image.load_img('test/sample.jpg', target_size=(224,224))
# prediction = model.predict(img)
# print prediction


